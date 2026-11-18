import argparse
import contextlib
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import utils
import trainer
import datasets
from models.digits import set_in_channels
from datasets.RflyMAD import (DATA_ROOT_DEFAULT as RFLYMAD_ROOT,
                              set_domain_mode, domain_pairs, domain_label)
from datasets.RflyMAD1 import (DATA_ROOT_DEFAULT as RFLYMAD1_ROOT,
                               set_domain_mode as set_domain_mode1,
                               domain_pairs as domain_pairs1,
                               domain_label as domain_label1)
from Batch_testing import evaluate


class _Tee:
    def __init__(self, stream, fp):
        self._stream = stream
        self._fp = fp

    def write(self, data):
        self._stream.write(data)
        self._fp.write(data)
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        finally:
            self._fp.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def tee_console(log_path):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8', buffering=1) as fp:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, fp), _Tee(old_err, fp)
        try:
            yield log_path
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            fp.flush()


def append_log(log_path, text):
    try:
        with open(log_path, 'a', encoding='utf-8') as fp:
            fp.write(text if text.endswith('\n') else text + '\n')
    except OSError as e:
        print('[warn] 追加日志失败: {} ({})'.format(log_path, e))


MAOSDAN_SHIFT_RECIPE = {

    'trans_weight': 1.0,
    'maosdan_shift_w': 0.5,

    'maosdan_shift_band_q': 1.0,
    'maosdan_shift_pull_frac': 0.5,
    'maosdan_shift_ev_closed': 0.45,
    'maosdan_shift_ev_knn': 0.45,
    'maosdan_shift_ev_geom': 0.10,
    'maosdan_shift_knn_k': 8,
    'maosdan_shift_bank_size': 1024,
    'maosdan_shift_min_side': 1,

    'maosdan_shift_dir': 'decision',
    'maosdan_shift_margin_target': 1.0,
    'maosdan_shift_eps_max': 2.0,
    'maosdan_shift_eps_rel': 0.5,

    'maosdan_shift_balance': 'reliability',

    'maosdan_shift_gap_scale': 0.15,
    'maosdan_shift_pull_w': None,
    'maosdan_shift_push_w': None,
    'maosdan_shift_balance_clip': 4.0,

    'maosdan_shift_proto_momentum': 0.9,
    'maosdan_shift_radius_q': 0.9,
    'maosdan_shift_radius_clip': 3.0,
    'maosdan_shift_min_support': 8,
    'maosdan_shift_start_epoch': 1,
    'maosdan_shift_orig_w': 0.0,
    'maosdan_shift_class_cap': 2.0,

    'maosdan_shift_gate_pull_geom': 1.0,
    'maosdan_shift_gate_push_geom': 0.0,
    'maosdan_shift_gate_pull_knn': 0.0,
    'maosdan_shift_gate_push_pu': 0.0,

    'maosdan_shift_margin_pull_max': 0.5,
    'maosdan_shift_pull_proto_agree': True,
    'maosdan_shift_ramp_epochs': 10,
    'maosdan_shift_coverage_ref': 0.10,

    'maosdan_shift_proto_gate_enable': 1,

}

_METHOD_RECIPES = {
    'maosdan_shift': MAOSDAN_SHIFT_RECIPE,
    'og_moe': MAOSDAN_SHIFT_RECIPE,
}


def apply_method_defaults(args):
    recipe = _METHOD_RECIPES.get(getattr(args, 'method', None))
    if not recipe:
        return args
    for key, value in recipe.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def build_parser():
    parser = argparse.ArgumentParser(
        description='开集域适应故障诊断：精简主入口（mfcl / maosdan_shift / og_moe）')
    parser.add_argument('--method', type=str, default='mfcl',
                        choices=['mfcl', 'maosdan_shift', 'og_moe'],
                        help='选择模型方法（精简版只保留这些）: '
                             'mfcl = MFCL；'
                             'maosdan_shift = MAOSDAN + 决策边界双向贴边修正；'
                             'og_moe = OG-MoE（MFCL-Lite + Boundary-Shift + learned gate，'
                             '单检查点；--og-shared 1 共享 backbone / 0 独立双 backbone）')
    parser.add_argument('--phase',               default='train',         type=str, help='train, test')
    parser.add_argument('--checkpoint',          default=None,            type=str,
                        help='test 阶段要加载的模型权重路径；缺省时用 {log}/checkpoints/ 下该方法的最佳检查点')
    parser.add_argument('--tsne-perplexity',     default=100,              type=int,
                        help='test 阶段 t-SNE 的困惑度')
    parser.add_argument('--tsne-iter',           default=1000,            type=int,
                        help='test 阶段 t-SNE 的迭代次数')
    parser.add_argument('--tsne-max-samples',    default=3000,            type=int,
                        help='test 阶段 t-SNE 的最大采样数（超过则随机子采样）')
    parser.add_argument('--data_name', type=str, default='RflyMAD',
                        choices=sorted(DATASET_PRESETS.keys()),
                        help='数据集名；同时决定默认的数据目录 / 域下标 / 类列表 / 输入通道数。'
                             'SDUST=轴承；RflyMAD=原版（case_split 前半/后半，域差 0.318）；'
                             'RflyMAD1=RflyMAD 的低域差重切分（同状态内 case 奇偶交替，'
                             '域差 0.165，默认域对 hover#b0->hover#b1）')

    parser.add_argument('--data_dir', type=str, default=None, help='the directory of the data')
    parser.add_argument('--src_name', type=str, default=None, help='源域下标（SDUST 工况号 / RflyMAD 域下标）')
    parser.add_argument('--tgt_name', type=str, default=None, help='目标域下标')
    parser.add_argument('--domain-mode', type=str, default=None,
                        choices=['flight', 'case_split', 'interleave'],
                        help='RflyMAD 专用：域的定义方式。flight=飞行状态（域下标 0..4）；'
                             'case_split=同一飞行状态内的 case 批次（域下标 = state*2+段号，'
                             '见 datasets/RflyMAD.py 第 1b 节）。缺省 = 用该模块里的默认值。'
                             'interleave 仅对 RflyMAD1 有效（同状态内 case 序号奇偶交替，默认）。')
    parser.add_argument('--src_classes', type=str, default=None, help='已知类列表（必须是 {0..K-1} 前缀）')
    parser.add_argument('--tgt_classes', type=str, default=None, help='目标域出现的全部类列表')
    parser.add_argument('--in-ch', dest='in_ch', type=int, default=None,
                        help='1D backbone 第一层的输入通道数（SDUST=1 / RflyMAD=34）；缺省按数据集预设')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='lenet',
                        choices=utils.get_model_names(),help='backbone architecture: ' + ' | '.join(utils.get_model_names()) +' (default: resnet18)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--stage2-lr', default=0.0001, type=float,
                        help='进入 stage2（epoch > --pretrain-epoch）后使用的**绝对**基学习率，'
                             '默认 1e-4（即默认启用）；衰减形状（--lr-gamma / --lr-decay）'
                             '不变，只换基值。想恢复改动前的行为就传与 --lr 相同的值')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50 , type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrain-epoch', default=4, type=int, help='pretrain epoch for discriminative feature learning')
    parser.add_argument('--maosdan-shift-w', default=None, type=float,
                        help='[maosdan_shift] **总开关 / 总权重 w0**（配方 0.5；0 = 完全关闭）。\n'
                             '0 时模型所有触点早退、shift_loss 返回 None ⇒ loss 表达式与\n'
                             '日志均与 --method maosdan **逐位等价**（消融口径）。\n'
                             '实际两侧权重：balance=prior 时 λ_pull = 2·w0·(1−π_u)、\n'
                             'λ_push = 2·w0·π_u（总强度恒为 2·w0，只是**在两侧之间重新分配**）。')
    parser.add_argument('--maosdan-shift-band-q', default=None, type=float,
                        help='[maosdan_shift] 边界带的覆盖比例（配方 0.5）：取判决余量 |m|\n'
                             '**最小的该比例**样本作为"容易跨界的样本"。\n'
                             '⚠️ 用**分位**而不是绝对阈值：绝对阈值是相位相关的（本仓库在\n'
                             'TGA 上实测过：训练早期源特征未聚好时 gate=0.8 也会放行 84%% 样本）。\n'
                             '经验：实测 src8 已知侧 P(unk) 的 p90 = 0.697 ⇒ m ≈ 0.83，\n'
                             '所以带宽必须够大才能盖住被过拒的已知样本 ⇒ 默认取 50%%。')
    parser.add_argument('--maosdan-shift-pull-frac', default=None, type=float,
                        help='[maosdan_shift] 带内 pull 的占比（配方 0.5 = 等量劈分）。\n'
                             '调大偏 pull（更保守、更像 TGA）；调小偏 push（更像 anomaly-mix）。\n'
                             '**等量劈分是本设计的关键**：两侧样本数相当 ⇒ 不产生净拒识压力，\n'
                             '从而避开本仓库所有单向力机制（openness / anomaly-mix / syn-unknown）\n'
                             '共同的失效模式"一端涨一端塌"。')
    parser.add_argument('--maosdan-shift-dir', default=None, type=str,
                        choices=['decision', 'geom'],
                        help='[maosdan_shift] 扰动方向（配方 decision）：\n'
                             'decision（默认）：u ∝ W_c − W_unk（head 权重行之差）。head 线性 ⇒\n'
                             '  d m / d z = W_unk − W_c，沿此方向走 ε 时 margin 的变化**精确**等于\n'
                             '  ε·‖W_c − W_unk‖ ⇒ 可用闭式步长让样本**恰好跨过边界**；\n'
                             '  ⚠️ 因为 u 是**判决空间**的方向，这是唯一"保证改判决"的选项。\n'
                             'geom：u ∝ μ_c − z（拉向类原型）/ z − μ_c*（远离最近原型），\n'
                             '  步长 = eps_rel·距离。更贴近"往已知类中心拉"的字面表述，但只改\n'
                             '  **特征几何**、不保证改判决（MGPL 就是"AUC 涨 0.039、oracle_H 只\n'
                             '  涨 0.0015"倒在这上面）——保留作消融对照。')
    parser.add_argument('--maosdan-shift-margin-target', default=None, type=float,
                        help='[maosdan_shift] decision 方向的目标余量 δ（配方 1.0）：\n'
                             'eps_pull = max(0, δ + m)/‖W_c−W_unk‖、eps_push = max(0, δ − m)/‖…‖\n'
                             '⇒ 扰动后的虚拟样本被放在**正确一侧、距边界 δ** 的位置上。\n'
                             'δ 越大 = 跨界走得越深（梯度信号更强，但越偏离"贴边"的初衷）。\n'
                             '样本已经站在正确一侧且足够深（|m| ≥ δ）时 eps 自动为 0 ⇒\n'
                             '力自动集中在真正贴边的样本上，安全样本零扰动。')
    parser.add_argument('--maosdan-shift-eps-max', default=None, type=float,
                        help='[maosdan_shift] 特征位移的硬上限 eps_max（配方 2.0）。\n'
                             '闭式步长与 ‖W_c − W_unk‖ 成反比：‖W‖ 小的时候 eps 会变得很大，\n'
                             '超出"局部线性有效"的范围 ⇒ 必须截断（截断后该样本只是没被完全\n'
                             '推过边界，不会造成错误方向）。')
    parser.add_argument('--maosdan-shift-eps-rel', default=None, type=float,
                        help='[maosdan_shift] geom 方向的相对步长（配方 0.5）：'
                             'eps = eps_rel × 到原型的距离，再被 eps_max 截断。'
                             '对 dir=decision 无效。')
    parser.add_argument('--maosdan-shift-balance', default=None, type=str,
                        choices=['reliability', 'gate', 'prior', 'fixed'],
                        help='[maosdan_shift] 两侧权重的分配方式（配方 **reliability**）：\n'
                             'reliability（默认）：**两侧同时开**（λ_pull = λ_push = w0），由样本级\n'
                             '  证据门槛各自挑安全候选（高证据的 m>0 样本 pull、低证据的 m<0 样本\n'
                             '  push），同一 batch 内两侧允许并存；\n'
                             '  ⚠️ 本模式**不做**"两侧等量"、也**不做**单侧门控：两侧都开，纯度\n'
                             '  由样本级证据门槛各自负责。单向消融用 `--maosdan-shift-pull-w 0` /\n'
                             '  `--maosdan-shift-push-w 0`。\n'
                             'gate（消融）：**免标签单侧门控** —— 用本模型对目标域的"预测未知率"\n'
                             '  EMA r 与 pi_u 的差决定只往哪一边发力：\n'
                             '    r > pi_u（过拒）=> 只 pull；r < pi_u（欠拒）=> 只 push；\n'
                             '    λ = w0·min(1, |r-pi_u| / --maosdan-shift-gap-scale)。\n'
                             '  ⚠️ 1200 目标样本实测：src8 的 m<0 池子 **96%% 是真已知**（push 会推掉\n'
                             '    456 个真已知）、src2 的 m>0 池子 **97%% 是真未知**（pull 会拉住 461\n'
                             '    个真未知）=> 低 pi_u 只有 pull 有意义、高 pi_u 只有 push 有意义\n'
                             '    （推算 ΔH 分别为 -0.045 / -0.026）；但它在 src8/src2 上只触发\n'
                             '    1~12/50 epoch，剂量过小。\n'
                             'prior：λ_pull=2w0(1-pi_u)、λ_push=2w0·pi_u（旧口径，留作消融）；\n'
                             'fixed：两侧都是 w0（消融）。')
    parser.add_argument('--maosdan-shift-gap-scale', default=None, type=float,
                        help='[maosdan_shift] gate 模式的**满强度偏差**（配方 0.15）：\n'
                             'gate = min(1, |r - pi_u| / scale)。实测 r：src8 0.383(pi_u .2, 偏差 .183)、\n'
                             'src2 0.709(pi_u .8, 偏差 .091) => 0.15 时分别是 1.00 / 0.61 的强度。\n'
                             '调大 = 更保守（要明显失准才发力）；调小 = 更激进。')
    parser.add_argument('--maosdan-shift-pull-w', default=None, type=float,
                        help='[maosdan_shift] 显式覆盖 λ_pull（默认 None = 由 balance 推）。\n'
                             '➔ 单向消融：`--maosdan-shift-push-w 0` = pull-only、\n'
                             '  `--maosdan-shift-pull-w 0` = push-only。')
    parser.add_argument('--maosdan-shift-push-w', default=None, type=float,
                        help='[maosdan_shift] 显式覆盖 λ_push（默认 None = 由 balance 推）。')
    parser.add_argument('--maosdan-shift-ev-closed', default=None, type=float,
                        help='[maosdan_shift] 证据分里**闭集置信**的权重（配方 0.45）。\n'
                             '依据：实测闭集 acc 0.878/0.901 明显高于 OS* 0.699/0.751 ⇒\n'
                             '被过拒的已知样本在"前 K 列 argmax"下大多是**对的**。')
    parser.add_argument('--maosdan-shift-ev-knn', default=None, type=float,
                        help='[maosdan_shift] 证据分里**目标域邻域一致性**的权重（配方 0.45）：\n'
                             '样本与上一个 epoch 目标域特征库里 k 近邻的"被判未知"平均概率取负。\n'
                             '这是全套可用信号里唯一**与 head 判决正交**的（用的是无标签的目标域\n'
                             '流形结构，不是源域统计量）；视图一致性未启用（用户指定），\n'
                             '所以它是承重的正交项。')
    parser.add_argument('--maosdan-shift-ev-geom', default=None, type=float,
                        help='[maosdan_shift] 证据分里**几何**的权重（配方 **0.10**，故意压低）。\n'
                             '⚠️ 为什么不能让它说了算：实测几何门控 "d/R≤1" 的逐类通过率\n'
                             'c6=0.90~0.96 而 c7=0.05~0.09、c0=0.10~0.13、c4=0.13~0.15，\n'
                             '但同一批样本 argmax(已知列)==c7 高达 0.91 ⇒ **几何说"未知"、\n'
                             'head 说"c7"，而它是真已知**。TGA 就是因为在几何门控上做正锚\n'
                             '而得到 −0.0166 的。这里几何只作弱证据项，绝不单独决定样本归属。')
    parser.add_argument('--maosdan-shift-knn-k', default=None, type=int,
                        help='[maosdan_shift] 邻域一致性的 k（配方 8）')
    parser.add_argument('--maosdan-shift-bank-size', default=None, type=int,
                        help='[maosdan_shift] 目标域特征库容量（配方 1024）。\n'
                             '库是**上一个 epoch** 观测的确定性等距抽样 ⇒ 不会"自己看自己"、\n'
                             '也不消耗全局 RNG（同 seed 的 ON/OFF 批序不分叉）。')
    parser.add_argument('--maosdan-shift-balance-clip', default=None, type=float,
                        help='[maosdan_shift] 类平衡权重的裁剪上限（配方 4.0）：'
                             '逐类逆频率加权后裁剪到最小权重的该倍数，再把权重和归一化到 1'
                             '（与 TGA / mfcl_known_stable 同口径）。')
    parser.add_argument('--maosdan-shift-proto-momentum', default=None, type=float,
                        help='[maosdan_shift] 源域原型的 EMA 动量（配方 0.9）：'
                             'μ_c ← 0.9·μ_c^new + 0.1·μ_c^old（首轮直接取均值）。')
    parser.add_argument('--maosdan-shift-radius-q', default=None, type=float,
                        help='[maosdan_shift] 类内半径的分位数（配方 0.9）：'
                             'R_c = 类内到 μ_c 的余弦距离的 0.9 分位（只用于证据分的几何项）。')
    parser.add_argument('--maosdan-shift-radius-clip', default=None, type=float,
                        help='[maosdan_shift] 半径裁剪倍数（配方 3.0）：'
                             'R_c ≤ 3×中位数，抑制重尾离群点把半径撑大（与 mm2/MGPL 同口径）。')
    parser.add_argument('--maosdan-shift-min-support', default=None, type=int,
                        help='[maosdan_shift] 逐类原型/半径的最少源域支持样本数（配方 8）。')
    parser.add_argument('--maosdan-shift-min-side', default=None, type=int,
                        help='[maosdan_shift] 每一侧的最少样本数（配方 3）：该 step 带内样本数\n'
                             '低于 2×该值时直接跳过（避免极少数样本拿满权重把早期表示带偏，\n'
                             '与 TGA 的 min_gated / cov_cap 同一个启机）。')
    parser.add_argument('--maosdan-shift-start-epoch', default=None, type=int,
                        help='[maosdan_shift] 从第几个 epoch 开始介入（配方 1）。\n'
                             '默认 1 而不是 0：源域原型与目标域特征库都要在 epoch 0 末才建好。\n'
                             '⚠️ 不要把"介入太晚"当小事：本仓库实测 msp 的 warmup=10 导致机制\n'
                             '完全没进上报指标（best 检查点落在 epoch 10 附近）。')
    parser.add_argument('--maosdan-shift-class-cap', default=None, type=float,
                        help='[maosdan_shift] **选择阶段**的每类覆盖上限 κ（配方 2.0）：\n'
                             '每类最多保留 ceil(κ·n_sel/K) 个候选；κ=1.0 = 严格均等，<=0 = 关闭。\n'
                             '⚠️ 逆频率损失权重（balance_clip）**改不了候选集合本身**：实测 src8 单个\n'
                             '  epoch 的 pull 池为 [3,0,0,1875,4,0,4,3]（K=8），即使做过裁剪，\n'
                             '  大类仍占总梯度质量的 97.1%% ⇒ 等效于秩 1 力。必须在选择阶段就限制覆盖。')
    parser.add_argument('--maosdan-shift-gate-pull-geom', default=None, type=float,
                        help='[maosdan_shift] pull 的**绝对门**（配方 1.0）：dmin/R_c <= 该值。\n'
                             'R_c 是**源域**类内 0.9 分位半径 ⇒ 1.0 就是源域类的边界，\n'
                             '无标签泄漏、无批内相对性。实测纯度 0.264→0.957 / 0.176→0.824 /\n'
                             '0.510→0.957 / 0.184→1.000（覆盖率降至 0.4~1.5/批）。<=0 = 关闭。')
    parser.add_argument('--maosdan-shift-gate-push-geom', default=None, type=float,
                        help='[maosdan_shift] push 的绝对门（配方 **0.0 = 关闭**）：dmin/R_c >= 该值。\n'
                             '⚠️ 实测 5 个检查点上**任何**门限的 push 池纯度都 <=0.63\n'
                             '（src8 三个臂全部 0.00~0.07，即池子几乎全是真已知）\n'
                             '⇒ 按"必须先验证各自可靠性"的要求默认关闭；仅用于消融。')
    parser.add_argument('--maosdan-shift-gate-pull-knn', default=None, type=float,
                        help='[maosdan_shift] pull 的附加门（配方 0.0 = 关闭）：\n'
                             '邻域判未知率 <= 该值。开启后纯度上升但覆盖率骤降至 ~0.1/批。')
    parser.add_argument('--maosdan-shift-gate-push-pu', default=None, type=float,
                        help='[maosdan_shift] push 的附加门（配方 0.0 = 关闭）：P(未知) >= 该值。')
    parser.add_argument('--maosdan-shift-margin-pull-max', default=None, type=float,
                        help='[maosdan_shift] pull 的**贴边门**（配方 0.5；<=0 = 关闭）：\n'
                             '只允许判决余量 m <= 该值的样本被 pull，其中 m = l_unk - max_c l_c。\n'
                             '动机（实测）：pull 池里 74%%~82%% 是真未知，而 m 大的样本就是那些\n'
                             '"明显未知" ⇒ 把它们拉回已知是纯伤害。密度 (0.5) 只留骑在边界上的样本。')
    parser.add_argument('--maosdan-shift-pull-proto-agree', default=None, type=int,
                        help='[maosdan_shift] pull 的**语义一致门**（配方 1 = 开启）：\n'
                             '要求 head 闭集 argmax（chat）等于最近源域原型类（cstar）。\n'
                             '不等时两个伪标签来源互相矛盾（head 说 c2、几何说 c7）⇒ 剔除。\n'
                             '与 preserve_class 互补：后者用位移**后**的标签，这里用原点的一对判决。')
    parser.add_argument('--maosdan-shift-ramp-epochs', default=None, type=int,
                        help='[maosdan_shift] λ 的**线性爬坡** epoch 数（配方 10；<=0 = 关闭）：\n'
                             '第 start_epoch 轮只有 1/ramp 的强度，逐 epoch 涨到 1.0。\n'
                             '动机（实测）：src4/seed202403 在 epoch1 一次性 pull 1575 个，\n'
                             'H 从 0.61 砸到 0.15 且永久停在坏 basin。')
    parser.add_argument('--maosdan-shift-coverage-ref', default=None, type=float,
                        help='[maosdan_shift] **候选覆盖率**参考值（配方 0.10；<=0 = 关闭）：\n'
                             '每步入选数/B 小于该值时，λ 按比例降权，\n'
                             '使 1~2 个候选样本拿不到与满支撑 batch 相同的整步力度。')
    parser.add_argument('--maosdan-shift-proto-gate-enable', default=None, type=int,
                        choices=[0, 1],
                        help='[maosdan_shift][消融专用] 原型可靠性筛选总开关（配方 1 = 启用）。\n'
                             '1 = 现行行为：候选必须过源域原型绝对门 g=dmin/R_c 与原型语义一致门；\n'
                             '0 = **只**去掉原型可靠性筛选，Pull / Push 两侧**照常开启**。\n'
                             '⚠️ 不要用 `--maosdan-shift-gate-pull-geom 0` 做这件事：旧代码里\n'
                             '   它等价于"把整个 pull 分支关掉"，会把两个消融混为一谈。')
    parser.add_argument('--ckpt-select-report-last', default=0, type=int, choices=[0, 1],
                        help='1 = 训练结束后**额外**用最后一个 epoch 的检查点（免标签选择）在目标域\n'
                             '再评估一次并打印 [label-free] 行；0（默认）= 与改动前逐位等价。\n'
                             '用于核对"按 val 目标域 H-score 选最佳检查点"这一规则是否影响结论。')
    parser.add_argument('--maosdan-shift-orig-w', default=None, type=float,
                        help='[maosdan_shift] 对**可靠 pull 样本**的未扰动特征额外施加一份 CE 的\n'
                             '权重（配方 0.10；0 = 不加）。'
                             '非 0 时该机制同时含一部分"带类平衡的伪标签 CE"，可用作对照。')
    parser.add_argument('--maosdan-geo-cov-cap', default=None, type=float,
                        help='[maosdan_stable_plus] 每步**覆盖率上限**（默认 0.25）：只取 s 最小的'
                             '该比例样本。这是必需的——绝对阈值是相位相关的，训练早期 R_c 松、'
                             's_p50 只有 0.47，即使 gate=0.8 也会放行 cov=0.84（含大量真未知），'
                             '导致 UK 塔缩。实测：OFF H=0.792 vs gate0.8/cov0.84 H=0.686'),
    parser.add_argument('--trans-weight', default=None, type=float,
                        help='[maosdan_adaptive / maosdan_openness] OSBP 边界项 trans_loss 的**基线**权重\n'
                             '（maosdan 基线 1.0）。\n'
                             '· maosdan_adaptive：作为自适应比例控制的锚点 base，'
                             'adaptive_open_set_balance() 每轮都以它重算，而不是在上一轮'
                             '结果上累乘；fixed 口径可传 --adaptive-osda 0 锁死为固定权重。\n'
                             '· maosdan_openness：作为 open_weight 的乘子底座，'
                             '有效权重 = 本值 × 单侧补偿权重（pi_u<=0.5 时为 1.0，'
                             'pi_u>0.5 时为 min(1+alpha*(pi_u-0.5), --maosdan-open-max)）。')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=202404, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--log-root', type=str, default='logs_main2',
                        help='各状态的训练/日志输出根目录（默认 logs_main2）。'
                             '消融实验可用它把不同配置的 checkpoint / txt 日志隔离到不同目录，避免互相覆盖')
    parser.add_argument('--states', nargs='+', default=None,
                        help='只跑指定条件（逐个匹配 STATES 的 key，例如 --states "src4(known4)" '
                             '"src3(known3)"）；缺省 = 跑 STATES 里的全部条件')
    parser.add_argument('--ensemble-with', dest='ensemble_with', default=None,
                        choices=['none', 'mfcl', 'maosdan_shift'],
                        help='[判决集成] 与另一个已训练好的方法做 softmax 概率融合（只影响 evaluate，不重训）：'
                             'p = w*softmax(本方法) + (1-w)*softmax(ensemble_with)。'
                             '不传 = 单模型，逐位不变。')
    parser.add_argument('--ensemble-w', dest='ensemble_w', default='0.3', type=str,
                        help='[判决集成] 本方法概率的权重 w：浮点数（默认 0.3），或 auto = '
                             '按 openness 自适应（π_u≤0.45→0.1，0.45~0.55→0.5，≥0.55→0.9）。'
                             'π_u 只用类别数算出、不读标签。RflyMAD1 用固定 0.3；'
                             'SDUST 的 mfcl/shift 强弱随 π_u 反转，用 auto。')
    parser.add_argument('--ensemble-log-root', dest='ensemble_log_root',
                        default='logs_rf1_methods', type=str,
                        help='[判决集成] 另一个方法的日志根目录（其下为 <method>/<state>/checkpoints/）；'
                             '默认 logs_rf1_methods，即 mfcl 的检查点位置。')
    parser.add_argument('--og-shared', dest='og_shared', default=1, type=int,
                        choices=[0, 1],
                        help='[og_moe] 1 = 两个专家共享 backbone（默认）；0 = 独立双 backbone')
    parser.add_argument('--og-expert-a', dest='og_expert_a', default='full',
                        choices=['full', 'lite'], type=str,
                        help='[og_moe] Expert A 的损失：full = 完整 MFCL'
                             '（sce+esl+csl+0.5*(ctl+prt+adv)）；lite = 旧试点配方'
                             '（无 CSL/ADV，仅用于复现）')

    parser.add_argument('--amp', default=0, type=int, choices=[0, 1],
                        help='是否启用自动混合精度(AMP)，1=开启')
    return parser


STATES = {

    'src6(known6)': [0, 1, 2, 3, 4, 5],
    'src9(known9)': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'src8(known8)': [0, 1, 2, 3, 4, 5, 6, 7],
    'src7(known7)': [0, 1, 2, 3, 4, 5, 6],

    'src5(known5)': [0, 1, 2, 3, 4],
    'src4(known4)': [0, 1, 2, 3],
    'src3(known3)': [0, 1, 2],
    'src2(known2)': [0, 1],
}


STATE_ARG_OVERRIDES = {}


RFLYMAD_STATES = {
    'K6(pi_u=0.143)': [0, 1, 2, 3, 4, 5],
    'K5(pi_u=0.286)': [0, 1, 2, 3, 4],
    'K4(pi_u=0.429)': [0, 1, 2, 3],
    'K3(pi_u=0.571)': [0, 1, 2],
    'K2(pi_u=0.714)': [0, 1],
    'K1(pi_u=0.857)': [0],
}


RFLYMAD1_STATES = {
    'K6(pi_u=0.143)': [0, 1, 2, 3, 4, 5],
    'K5(pi_u=0.286)': [0, 1, 2, 3, 4],
    'K4(pi_u=0.429)': [0, 1, 2, 3],
    'K3(pi_u=0.571)': [0, 1, 2],
    'K2(pi_u=0.714)': [0, 1],
}


_DATA_DIR_DEFAULT = os.environ.get(
    'DVC_HE_DATA_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

DATASET_PRESETS = {
    'SDUST': {
        'data_dir': _DATA_DIR_DEFAULT,
        'src_name': [2], 'tgt_name': [3],
        'src_classes': [0, 1, 2, 3, 4, 5, 6],
        'tgt_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'in_ch': 1,
    },
    'RflyMAD': {

        'data_dir': RFLYMAD_ROOT,
        'src_name': [2], 'tgt_name': [3],
        'src_classes': [0, 1, 2, 3, 4, 5],
        'tgt_classes': [0, 1, 2, 3, 4, 5, 6],

        'in_ch': 30,
    },
    'RflyMAD1': {

        'data_dir': RFLYMAD1_ROOT,
        'src_name': [0], 'tgt_name': [1],
        'src_classes': [0, 1, 2, 3, 4, 5],
        'tgt_classes': [0, 1, 2, 3, 4, 5, 6],
        'in_ch': 30,
    },
}


def states_for(data_name):
    name = str(data_name).lower()
    if name == 'rflymad1':
        return RFLYMAD1_STATES
    if name == 'rflymad':
        return RFLYMAD_STATES
    return STATES


def resolve_dataset_defaults(args):
    preset = DATASET_PRESETS.get(str(args.data_name))
    if preset is None:
        raise SystemExit('未知 --data_name={}（可选：{}）'.format(
            args.data_name, list(DATASET_PRESETS)))
    filled = []
    for key, value in preset.items():
        if key == 'in_ch':
            continue
        if getattr(args, key, None) is None:
            setattr(args, key, value)
            filled.append(key)
    if getattr(args, 'in_ch', None) is None:
        args.in_ch = preset['in_ch']
        filled.append('in_ch')

    set_in_channels(args.in_ch)

    if str(args.data_name) == 'RflyMAD1':
        if getattr(args, 'domain_mode', None) not in (None, 'interleave'):
            raise SystemExit(
                "RflyMAD1 的域定义固定为 interleave（同状态内 case 序号交错分组），"
                "不支持 --domain-mode={}。需要按飞行状态跨域请用 --data_name RflyMAD "
                "--domain-mode flight。".format(args.domain_mode))
        set_domain_mode1(None)
        dm = 'interleave'
        label_fn, pairs_fn = domain_label1, domain_pairs1
    else:
        dm = set_domain_mode(getattr(args, 'domain_mode', None))
        label_fn, pairs_fn = domain_label, domain_pairs
    if str(args.data_name) in ('RflyMAD', 'RflyMAD1'):
        def _first(v):
            return int(v[0] if isinstance(v, (list, tuple)) else v)
        print('[域定义] dataset={} domain_mode={} | src={}(#{}) -> tgt={}(#{}) | 预置域对={}'
              .format(args.data_name, dm, label_fn(_first(args.src_name)), _first(args.src_name),
                      label_fn(_first(args.tgt_name)), _first(args.tgt_name),
                      [(a, b) for a, b in pairs_fn()]), flush=True)
    print('[数据集] {} | auto-filled: {} | data_dir={} | src_name={} tgt_name={} | '
          'src_classes={} tgt_classes={} | in_ch={}'.format(
              args.data_name, filled or '(none)', args.data_dir,
              args.src_name, args.tgt_name, args.src_classes,
              args.tgt_classes, args.in_ch), flush=True)
    return args


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _log_name(args):
    suffix = '_test' if str(getattr(args, 'phase', 'train')).lower() == 'test' else ''
    return '{}_{}{}.txt'.format(args.method, args.epochs, suffix)


def run_state(name, src_classes, base_args, state_index):
    args = copy.deepcopy(base_args)
    args.src_classes = list(src_classes)

    for key, value in STATE_ARG_OVERRIDES.get(name, {}).items():
        setattr(args, key, value)
    set_seed(args.seed + state_index)

    args.log = os.path.join(getattr(args, 'log_root', 'logs_main2'), name)
    os.makedirs(os.path.join(args.log, 'checkpoints'), exist_ok=True)

    log_txt = os.path.join(args.log, _log_name(args))

    with tee_console(log_txt):
        print('=' * 72)
        print('日志文件 : {}'.format(os.path.abspath(log_txt)))
        print('模型方法 : {}    epoch 数 : {}'.format(args.method, args.epochs))
        print('状态     : {}    seed : {}'.format(name, args.seed))
        print('开始时间 : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        print('=' * 72)

        data_loaders = getattr(datasets, args.data_name)
        try:
            loaders = data_loaders(args.data_dir, args.src_name, args.tgt_name,
                                   list(args.src_classes), args.tgt_classes, 32, 4).create_loaders()
        except TypeError:
            loaders = data_loaders(args.data_dir, args.src_name, args.tgt_name,
                                   list(args.src_classes), args.tgt_classes, 32).create_loaders()
        (train_source_loader, train_target_loader,
         val_source_loader, val_target_loader, num_classes) = loaders

        n_known = len(src_classes)
        print('\n################ 状态 {}: src_classes={} (已知{}类/未知{}类) num_classes={}'
              .format(name, src_classes, n_known, num_classes - n_known, num_classes), flush=True)

        t0 = time.time()
        trainer.main(args)
        print('  [{}] 训练完成，耗时 {:.1f}s'.format(name, time.time() - t0), flush=True)

        m = evaluate(args, val_target_loader, num_classes, src_loader=val_source_loader)
        print('  [{}] OS*={:.4f} UK={:.4f} OS={:.4f} H={:.4f}'
              .format(name, m['os_star'], m['uk'], m['os'], m['h']), flush=True)
        if 'h_mgpl' in m:
            print('  [{}] [MGPL J={} gamma={:.3f}]  OS*={:.4f} UK={:.4f} H={:.4f}'
                  '   (同权重 head 口径 H={:.4f}；oracle-H 上界={:.4f}；'
                  'ratio 中位 源={:.3f} 目标={:.3f})'.format(
                      name, int(m['mgpl_J']), m['mgpl_gamma'], m['os_star_mgpl'],
                      m['uk_mgpl'], m['h_mgpl'], m['h'], m['oracle_h_mgpl'],
                      m['ratio_src_p50'], m['ratio_tgt_p50']), flush=True)
            print('  [{}] [MGPL pi_u={:.1f} 匹配]  d_min: OS*={:.4f} UK={:.4f} H={:.4f} '
                  'pred_unk={:.3f} (theta={:.3f}) | ratio: H={:.4f}'
                  '   [oracle 上界 d_min={:.4f} ratio={:.4f}]'.format(
                      name, m['pi_u'], m['os_star_dmin_piu'], m['uk_dmin_piu'],
                      m['h_dmin_piu'], m['pred_unk_dmin_piu'], m['theta_dmin_piu'],
                      m['h_ratio_piu'], m['oracle_h_dmin'], m['oracle_h_mgpl']),
                  flush=True)
        if 'nci_h' in m:
            print('  [{}] [NCI alpha={:.1e} q={:.2f}]  OS*={:.4f} UK={:.4f} OS={:.4f} H={:.4f}'
                  '   (pred_unk={:.3f}，tau={:.5f}；AUROC={:.4f}；oracle-H={:.4f}；'
                  's: 源={:.3f} 目标={:.3f} 已知={:.3f} 未知={:.3f}；编码自检 dH={:.1e})'
                  .format(name, m['nci_alpha'], m['nci_thr_q'], m['nci_os_star'], m['nci_uk'],
                          m['nci_os'], m['nci_h'], m['nci_pred_unk'], m['nci_tau'],
                          m['nci_auc'], m['nci_oracle_h'], m['nci_src_p50'], m['nci_tgt_p50'],
                          m['nci_s_known_p50'], m['nci_s_unknown_p50'],
                          abs(m['nci_h'] - m['nci_h_check'])),
                  flush=True)

        print('结束时间 : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))

    print('  [{}] 日志已写入: {}'.format(name, os.path.abspath(log_txt)), flush=True)
    return m


def main():
    base_args = build_parser().parse_args()

    resolve_dataset_defaults(base_args)

    apply_method_defaults(base_args)

    states = states_for(base_args.data_name)
    only = getattr(base_args, 'states', None)
    items = [(n, s) for n, s in states.items() if (not only or n in only)]
    if not items:
        raise SystemExit('[--states] 没有任何条件匹配：{}（可选：{}）'.format(
            only, list(states.keys())))
    print('本次运行的条件：{}'.format([n for n, _ in items]), flush=True)
    results = {}
    for state_index, (name, src_classes) in enumerate(items):
        results[name] = run_state(name, src_classes, base_args, state_index)

    summary = []
    summary.append('\n\n========== 最终结果汇总 ==========')
    summary.append('{:>14} | {:>10} | {:>10} | {:>10} | {:>10}'.format('状态', 'OS*', 'UK', 'OS', 'H'))
    summary.append('-' * 56)
    for name, m in results.items():
        summary.append('{:>14} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f}'.format(
            name, m['os_star'], m['uk'], m['os'], m['h']))
    summary.append('=' * 48 + '\n')
    summary_text = '\n'.join(summary)
    print(summary_text)

    log_root = getattr(base_args, 'log_root', 'logs_main2')
    for name in results:
        append_log(os.path.join(log_root, name, _log_name(base_args)), summary_text)


if __name__ == '__main__':
    main()
