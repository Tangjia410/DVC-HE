import os

import numpy as np


def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _import_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_matrix(y_true, y_pred, class_names, save_path,
                          normalize=False, title='Confusion Matrix',
                          cmap='Blues', dpi=200, figsize=None):
    plt = _import_plt()

    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    n = len(class_names)

    cm = np.zeros((n, n), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1.0

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm, np.maximum(row_sum, 1e-12))
        num_fmt = '{:.2f}'
    else:
        cm_show = cm
        num_fmt = '{:d}'

    if figsize is None:
        figsize = (max(6.0, 0.78 * n), max(5.0, 0.68 * n))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_show, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm_show.max() / 2.0 if cm_show.max() > 0 else 0.5
    font_sz = 8 if n > 12 else 10
    for i in range(n):
        for j in range(n):
            val = cm_show[i, j]
            ax.text(j, i, num_fmt.format(int(val) if num_fmt == '{:d}' else val),
                    ha='center', va='center', fontsize=font_sz,
                    color='white' if val > thresh else 'black')

    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return cm


def plot_tsne(features, labels, class_names, save_path, title='t-SNE',
              perplexity=30, n_iter=1000, random_state=42, dpi=200,
              max_samples=3000, unknown_index=None, figsize=(9, 7),
              pred_labels=None):
    plt = _import_plt()
    from sklearn.manifold import TSNE

    feats = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(int)
    n_orig = feats.shape[0]
    n = n_orig
    if n < 2:
        raise ValueError('t-SNE 至少需要 2 个样本，当前为 {}'.format(n))

    pred = None
    if pred_labels is not None:
        pred = np.asarray(pred_labels).reshape(-1).astype(int)
        if pred.shape[0] != n_orig:
            raise ValueError('pred_labels 长度 {} 与 features 的 {} 不一致'.format(
                pred.shape[0], n_orig))

    idx = None
    if n > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, max_samples, replace=False)
        feats = feats[idx]
        labels = labels[idx]
        if pred is not None:
            pred = pred[idx]
        n = max_samples

    perp = float(min(perplexity, max(2, (n - 1) // 3)))

    try:
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=n_iter,
                    random_state=random_state, init='pca')
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perp, max_iter=n_iter,
                    random_state=random_state, init='pca')
    emb = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=figsize)
    uniq = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap('tab20', max(len(uniq), 1))

    for k, c in enumerate(uniq):
        mask = labels == c
        if unknown_index is not None and c == unknown_index:
            name = 'Unknown'
        elif 0 <= c < len(class_names):
            name = str(class_names[c])
        else:
            name = str(c)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.75,
                   color=cmap(k), label='{} ({})'.format(name, int(mask.sum())))

    if pred is not None and unknown_index is not None:
        n_fr = n_fa = 0
        fr = (labels != unknown_index) & (pred == unknown_index)
        if fr.any():
            n_fr = int(fr.sum())
            ax.scatter(emb[fr, 0], emb[fr, 1], s=34, facecolors='none',
                       edgecolors='k', linewidths=0.7, marker='o',
                       label='False rejection ({})'.format(n_fr), zorder=3)
        fa = (labels == unknown_index) & (pred != unknown_index)
        if fa.any():
            n_fa = int(fa.sum())
            ax.scatter(emb[fa, 0], emb[fa, 1], s=26, facecolors='none',
                       edgecolors='k', linewidths=0.7, marker='s',
                       label='False acceptance ({})'.format(n_fa), zorder=3)
        print('[plot_tsne] overlay: false-rejection={} false-acceptance={} '
              '(of {} plotted)'.format(n_fr, n_fa, int(labels.shape[0])))

    ax.set_title(title)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(loc='best', fontsize='small', markerscale=1.5, ncol=2)
    fig.tight_layout()
    _ensure_dir(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return emb
