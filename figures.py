import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from statsmodels.distributions.empirical_distribution import ECDF
from functions import get_iat_broad


# Create ECDF Plots
def plot_ecdf_thresholds(data_frames,
                         labels,
                         category,
                         dtime,
                         thresholds):
    """
    data_frames: contains recommendations for each sample
    labels: array of class labels e.g. ['White', 'Non-White']
    category: 'Race', 'Age'
    dtime: DateTime
    thresholds: dict of each abx with correpsonding threshold
    """
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    data_list = data_frames.copy()
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    axes = [ax for ax in fig.axes]
    tup = zip(abxs, axes)
    colors = ['blue', 'darkorange', 'green']
    patches = []

    for abx, ax in tup:

        # Create ECDF for the antibiotic for each group
        lab = 'predicted_prob_' + abx
        for df, c, label in zip(data_list, colors[:len(data_list)], labels):
            if df.shape[0] == 0:
                continue
            ecdf = ECDF(df[lab])
            ax.plot(ecdf.x, ecdf.y, color=c)

        # Make plot
        ax.set_title(abx)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('% of Observations')
        ax.axvline(thresholds[abx], linestyle='--', color='black')
        ax.text(x=thresholds[abx] + 0.1,
                y=0.05,
                s='Threshold:' + str(round(thresholds[abx], 3)),
                size='large')
        ax.grid(b=True)

        # Subset dataframe to not include the prev antibiotic recs
        for i, df in enumerate(data_list):
            data_list[i] = df[df.rec_final != abx]

    # Create patches for legend
    for c, label in zip(colors, labels):
        p = mpatches.Patch(color=c, label=label)
        patches.append(p)

    # Adjust spacing
    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.90)

    # Add legend and title
    plt.legend(title=category, handles=patches)
    plt.suptitle('Cumulative Distributions With Optimal Thresholds Conditional on ' + category,
                 size='20')

    # Save figure in output folder
    plt.savefig(f"{dtime}/{dtime}" + '_ECDF_' + f'{category}'
                + '.png', dpi=300)


# Plots IAT vs Broad conditional on Race
def iat_broad_plot(data_frames, labels, category, dtime):
    # Get IAT and 2nd line usage prop for dataframes
    points = []
    for df in data_frames:
        points.append(get_iat_broad(df, 'rec_final'))

    plt.subplots(figsize=(10, 10))
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)

    # plt.plot([white_pt[1], n_white_pt[1]], [white_pt[0], n_white_pt[0]], marker='o',
    #          zorder=1, color='black', alpha=0.8)

    # Published result
    paper_pt = [0.11, 0.098]
    plt.scatter(paper_pt[0], paper_pt[1], color='grey', s=200, zorder=2)
    plt.text(paper_pt[0] + 0.005, paper_pt[1] + 0.01, 'Published Result', color='grey')

    colors = ['blue', 'orange', 'green']

    for pt, label, col in zip(points, labels, colors):
        plt.text(pt[1] + 0.005, pt[0] + 0.01, label, color=col)
        plt.scatter(pt[1], pt[0], color=col, s=200, zorder=2)

    plt.xlabel('% Broad Spectrum Antibiotic Use', size=16)
    plt.xticks(size=14)
    plt.ylabel('% Inappropiate Antibiotic Therapy Use', size=16)
    plt.yticks(size=14)
    plt.title(f'IAT vs % Broad Spectrum Use Conditional on {category}',
              size=20,
              pad=20)
    plt.savefig(f"{dtime}/{dtime}_iat_broad_{category}.png", dpi=300)


# Plots IAT and Broad conditional on Age
def iat_broad_plot_age(data_frames, time):
    pts = []
    for g in data_frames:
        pts.append((get_iat_broad(g, 'rec_final')[1], get_iat_broad(g, 'rec_final')[0]))
    paper_pt = [0.11, 0.098]
    plt.subplots(figsize=(10, 10))
    for i, k in enumerate(list(zip(['Group 0', 'Group 1', 'Group 2'], ['blue', 'darkorange', 'green']))):
        plt.scatter(pts[i][0], pts[i][1], marker='o', color=k[1], s=200)
        plt.text(pts[i][0] + 0.005, pts[i][1] + 0.01, s=k[0], color=k[1])
    plt.text(paper_pt[0] - 0.04, paper_pt[1] + 0.005, 'Published Result', color='grey')
    plt.scatter(paper_pt[0], paper_pt[1], color='grey', s=200, zorder=2)
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)
    plt.xlabel('% Broad Spectrum Antibiotic Use', size='large')
    plt.ylabel('% Inappropiate Antibiotic Therapy Use', size='large')
    plt.title("IAT vs % Broad Spectrum Use Conditional on Age Subgroup", size='x-large')
    plt.legend(labels=['18-27', '27-39', '>39'])
    plt.savefig(f"{time}/{time}" + "_iat_broad_age.png", dpi=300)


# Plots sensitivity analysis figure of IAT vs 2nd line usage
# Provide both groups of dataframes
def plot_thresholds_stats_by_race(w_df, nw_df, time):
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=(20, 10))
    wh = sns.scatterplot(data=w_df, x='broad_prop', y='iat_prop',
                         ax=ax, alpha=0.5, label='White')
    nwh = sns.scatterplot(data=nw_df, x='broad_prop', y='iat_prop',
                          ax=ax, alpha=0.5, label='Non-White')

    # plt.scatter(w_broad_prop, w_iat_prop, color='green')
    # plt.scatter(nw_broad_prop, nw_iat_prop, color='purple')
    plt.scatter(0.095977, 0.10184, color="yellow")  # Published result

    plt.title('IAT vs Broad for Subgroup Using Thresholds for All Patients', size=20)
    plt.legend(prop={'size': 20})
    plt.xlabel('% Broad Spectrum Use', size='x-large')
    plt.ylabel('% IAT', size='x-large')

    ax.axvline(x=0.1, color='black', alpha=0.5)
    ax.text(0.12, 0.07, '% Broad = 0.1', size='large')
    ax.plot()
    plt.savefig(f"{time}/{time}_iat_vs_broad_thresholds_race.png", dpi=300)
