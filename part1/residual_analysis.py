import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from part1.ols_implementation import _add_intercept


def residual_plots(X, y, beta_hat, save_path=None):
    """
    @brief  Vẽ 4 biểu đồ chẩn đoán phần dư (Residual Diagnostic Plots)
            để kiểm tra các giả thiết của mô hình hồi quy tuyến tính.

            Bao gồm:
            1. Residuals vs Fitted
            2. Normal Q-Q Plot
            3. Scale-Location
            4. Cook's Distance

    @input  X         : array-like, shape (n, p)
                        Ma trận đặc trưng (chưa có intercept).

            y         : array-like, shape (n,) hoặc (n, 1)
                        Vector giá trị thực tế.

            beta_hat  : np.ndarray, shape (p+1, 1)
                        Vector hệ số hồi quy.

            save_path : str hoặc None
                        Nếu khác None thì figure sẽ được lưu ra file.

    @output None.
    """

    def _running_mean(x, y, window):
        """
        Hàm nội bộ làm mượt đường xu hướng bằng running mean.
        """
        half = window // 2
        n_local = len(y)

        y_smooth = np.array([
            np.mean(
                y[max(0, i - half): min(n_local, i + half + 1)]
            )
            for i in range(n_local)
        ])

        return x, y_smooth

    # Chuẩn bị dữ liệu
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).reshape(-1, 1)
    X_int = _add_intercept(X_arr)
    n, p_plus_1 = X_int.shape
    p = p_plus_1 - 1

    # Fitted values
    y_hat = X_int @ beta_hat

    # Residuals
    residuals = y_arr - y_hat

    # RSS
    rss = np.sum(residuals ** 2)

    # Sigma^2
    df = n - p - 1

    if df <= 0:
        raise ValueError(
        "Số bậc tự do không hợp lệ."
    )
    sigma2 = rss / df

    sigma = np.sqrt(max(sigma2, 1e-12))

    # Hat matrix
    XTX_inv = np.linalg.inv(X_int.T @ X_int)

    H = X_int @ XTX_inv @ X_int.T

    h = np.diag(H)

    # Standardized residuals
    denom = sigma * np.sqrt(np.maximum(1 - h, 1e-10))

    std_residuals = residuals.flatten() / denom

    # Cook's Distance
    cooks_d = (
        (residuals.flatten() ** 2)
        / (p_plus_1 * sigma2)
    ) * (
        h / np.maximum((1 - h) ** 2, 1e-10)
    )

    y_hat_flat = y_hat.flatten()

    obs_index = np.arange(n)

    # Plot figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        'Residual Diagnostic Plots',
        fontsize=15,
        fontweight='bold'
    )

    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(
        y_hat_flat,
        residuals.flatten(),
        alpha=0.6,
        edgecolors='steelblue',
        facecolors='lightblue',
        s=40,
        linewidths=0.6
    )

    ax1.axhline(
        0,
        color='red',
        linestyle='--',
        linewidth=1.2
    )

    sort_idx = np.argsort(y_hat_flat)
    window = max(int(0.2 * n), 5)
    smooth_x, smooth_y = _running_mean(
        y_hat_flat[sort_idx],
        residuals.flatten()[sort_idx],
        window
    )

    ax1.plot(
        smooth_x,
        smooth_y,
        color='red',
        linewidth=1.5
    )

    ax1.set_title(
        'Residuals vs Fitted',
        fontweight='bold'
    )

    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.grid(True, alpha=0.3)

    # 2. Normal Q-Q Plot
    ax2 = axes[0, 1]
    (osm, osr), (slope, intercept, _) = scipy.stats.probplot(
        std_residuals,
        dist='norm',
        fit=True
    )

    ax2.scatter(
        osm,
        osr,
        alpha=0.6,
        edgecolors='steelblue',
        facecolors='lightblue',
        s=40
    )

    ax2.plot(
        osm,
        slope * np.array(osm) + intercept,
        color='red',
        linewidth=1.5
    )

    ax2.set_title(
        'Normal Q-Q Plot',
        fontweight='bold'
    )

    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Standardized Residuals')
    ax2.grid(True, alpha=0.3)

    # 3. Scale-Location
    ax3 = axes[1, 0]
    sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))

    ax3.scatter(
        y_hat_flat,
        sqrt_abs_std_res,
        alpha=0.6,
        edgecolors='steelblue',
        facecolors='lightblue',
        s=40
    )

    sort_idx2 = np.argsort(y_hat_flat)

    smooth_x2, smooth_y2 = _running_mean(
        y_hat_flat[sort_idx2],
        sqrt_abs_std_res[sort_idx2],
        window
    )

    ax3.plot(
        smooth_x2,
        smooth_y2,
        color='red',
        linewidth=1.5
    )

    ax3.set_title(
        'Scale-Location',
        fontweight='bold'
    )

    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.grid(True, alpha=0.3)

    # 4. Cook's Distance
    ax4 = axes[1, 1]
    threshold = 4 / n

    colors = np.where(
        cooks_d > threshold,
        'tomato',
        'lightblue'
    )

    edge_colors = np.where(
        cooks_d > threshold,
        'darkred',
        'steelblue'
    )

    ax4.bar(
        obs_index,
        cooks_d,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.5,
        alpha=0.8
    )

    ax4.axhline(
        threshold,
        color='red',
        linestyle='--',
        linewidth=1.2
    )

    influential = np.where(cooks_d > threshold)[0]

    for idx in influential:

        ax4.annotate(
            str(idx),
            xy=(idx, cooks_d[idx]),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            fontsize=7,
            color='darkred'
        )

    ax4.set_title(
        "Cook's Distance",
        fontweight='bold'
    )

    ax4.set_xlabel('Observation Index')
    ax4.set_ylabel("Cook's Distance")
    ax4.grid(True, alpha=0.3, axis='y')

    # Hiển thị / lưu figure
    plt.tight_layout()
    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=150,
            bbox_inches='tight'
        )

        plt.close(fig)

    else:
        plt.show()
