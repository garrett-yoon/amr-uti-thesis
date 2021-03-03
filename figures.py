import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functions import get_iat_broad


# Create ECDF Plots
def plot_ecdf_thresholds(data_list,
                         labels,
                         split_s,
                         time,
                         thresholds):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    data_list = data_list.copy()
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

    for c, label in zip(colors, labels):
        p = mpatches.Patch(color=c, label=label)
        patches.append(p)

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.90)
    plt.legend(title=split_s, handles=patches)
    plt.suptitle('Cumulative Distributions With Optimal Thresholds Conditional on ' + split_s,
                 size='20')
    plt.savefig(f"{time}/{time}" + '_ECDF_' + f'{split_s}'
                + '.png', dpi=300)


# Plots IAT vs Broad conditional on Race
def iat_broad_plot_race(dfs, time):
    white_pt = get_iat_broad(dfs[0], 'rec_final')
    n_white_pt = get_iat_broad(dfs[1], 'rec_final')
    plt.subplots(figsize=(10, 10))
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)
    plt.plot([white_pt[1], n_white_pt[1]], [white_pt[0], n_white_pt[0]], marker='o',
             zorder=1, color='black', alpha=0.8)

    paper_pt = [0.11, 0.098]
    plt.scatter(paper_pt[0], paper_pt[1], color='grey', s=200, zorder=2)
    plt.text(white_pt[1] + 0.005, white_pt[0] + 0.01, 'White', color='blue')
    # plt.scatter(all_pt[1], all_pt[0], color='grey', s=200,zorder=2)

    plt.scatter(white_pt[1], white_pt[0], color='blue', s=200, zorder=2)
    plt.scatter(n_white_pt[1], n_white_pt[0], color='orange', s=200, zorder=2)
    plt.text(white_pt[1] + 0.005, white_pt[0] + 0.01, 'White', color='blue')
    plt.text(n_white_pt[1] + 0.005, n_white_pt[0] + 0.01, 'Non-White', color='orange')
    plt.text(paper_pt[0] + 0.005, paper_pt[1] + 0.01, 'Published Result', color='grey')
    plt.xlabel('% Broad Spectrum Antibiotic Use', size=16)
    plt.xticks(size=14)
    plt.ylabel('% Inappropiate Antibiotic Therapy Use', size=16)
    plt.yticks(size=14)
    plt.title("IAT vs % Broad Spectrum Use Conditional on Racial Subgroup",
              size=20,
              pad=20)
    plt.savefig(f"{time}/{time}" + "_iat_broad_race.png", dpi=300)


# Plots IAT and Broad conditional on Age
def iat_broad_plot_age(dfs, time):
    pts = []
    for g in dfs:
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