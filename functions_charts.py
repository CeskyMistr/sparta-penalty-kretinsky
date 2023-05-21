import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma, kstest, beta
import plotnine as p9


def visualize_and_test_gamma_dist(dt, column):
    params = gamma.fit(dt[column])
    ks_statistic, p_value = kstest(dt[column], 'gamma', args=params)

    # Plot the histogram of the data and the fitted gamma distribution for all matches
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.hist(dt[column], bins=30, density=True, alpha=0.5)
    x = np.linspace(dt[column].min(), dt[column].max(), 100)
    ax.plot(x, gamma.pdf(x, *params), 'r-', lw=2, label='gamma pdf')
    # Add text to the plot
    textstr = '\n'.join([f"{param}: {val:.4f}" for param, val in zip(["a", "loc", "scale"], params)])
    textstr += f"\nK-S: {ks_statistic:.4f}\np-val: {p_value:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    # Add title
    ax.set_title(f"Gamma rozdělení pro intervence rozhodčích na počet útoků s K-S testem shody", fontsize=16)
    plt.savefig('gamma_plot.png')
    plt.show()


def plot_sampled_penalties(hist_data, file_name):
    # Create the bar chart
    plt.figure(figsize=(16, 9))
    plt.bar(hist_data.index, hist_data.values)

    # Color the bins above 6 in red
    red_threshold = 7
    plt.bar(hist_data.index[hist_data.index >= red_threshold],
            hist_data.values[hist_data.index >= red_threshold],
            color='red')

    # Set specific x-axis labels
    plt.xticks(hist_data.index, hist_data.index.astype(int).astype(str), fontsize=12)
    plt.yticks(fontsize=12)

    # Set labels and title
    plt.xlabel('Počet penalt', fontsize=14)
    plt.ylabel('Četnost (%)', fontsize=14)
    plt.title('Očekávaný počet penalt v 8 zápasech Sparty', fontsize=16)

    # Add legend
    plt.legend(['Méně než Sparta zahrávala', 'Stejně a více jako Sparta zahrávala'], loc='upper right', fontsize=12)

    # Add annotation with curved arrow
    annotation_text = f'Pravděpodobnost 7 a více penalt v 8 zápasech je {round(sum(hist_data.values[hist_data.index >= red_threshold]), 1)} %.'
    arrow_x = red_threshold
    arrow_y = 1.5
    plt.annotate(annotation_text, xy=(arrow_x+0.50, arrow_y), xytext=(arrow_x-0.75, arrow_y+5),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

    # Annotate bars with y-axis values
    for x, y in zip(hist_data.index, hist_data.values):
        plt.annotate(f'{np.round(y, 1)}', xy=(x, y), xytext=(x, y+0.1),
                    ha='center', va='bottom', fontsize=11)

    # Save the plot
    plt.savefig(file_name)
    # Show the plot
    plt.show()


def plot_posterior_distributions(alpha_a, beta_a, alpha_b, beta_b, prob_b_greater):

    # Generate x-axis values
    x = np.linspace(0, 1, 1000)

    # Compute the probability density function (PDF) for each distribution
    pdf_a = beta.pdf(x, alpha_a, beta_a)
    pdf_b = beta.pdf(x, alpha_b, beta_b)

    # Plot the distributions
    plt.figure(figsize=(16, 9))
    plt.plot(x, pdf_a, label='Před návštěvou')
    plt.plot(x, pdf_b, label='Po Křetínského návštěvě', color="red")
    plt.xlabel('Podíl zápasů s penaltou', fontsize=14)
    plt.ylabel('Hustota pravděpodobnosti', fontsize=14)
    plt.title('Podíl zápasů s penaltou Sparty před a po Křetínského návštěvě rozhodčích', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Add annotation
    plt.annotate(f'Podíl zápasů s penaltou\nje na {np.round(prob_b_greater * 100, 2)} % po Křetínského návštěvě větší.', xy=(x[np.argmax(pdf_b)], pdf_b.max() * 1.4),
                xytext=(x[np.argmax(pdf_b)], pdf_b.max() * 1.4), color='black',
                backgroundcolor='white', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

    # Save the plot
    plt.savefig("sparta_penalty_bayesian.png")
    # Show the plot
    plt.show()


def plot_coeff_plot(model, title, file_name):
    # Get the coefficient estimates and standard errors from the model
    coefs = model.params.reset_index()
    coefs.columns = ['variable', 'coefficient']
    coefs['std_error'] = model.bse.reset_index()[0]
    coefs["error_min"] = coefs["coefficient"] - 1.96 * coefs["std_error"]
    coefs["error_max"] = coefs["coefficient"] + 1.96 * coefs["std_error"]

    filtr = coefs["variable"].str.contains("kretinsky|Slavia|Sparta|hga")
    coefs = coefs.loc[filtr,:]

    var_names_mapper = {
    "team[T.Slavia Praha]": "Slavia",
    "team[T.Sparta Praha]": "Sparta",
    "team_opp[T.Slavia Praha]": "Slavia soupeř",
    "team_opp[T.Sparta Praha]": "Sparta soupeř",
    "kretinsky_sparta": "Sparta po Křetínském",
    "kretinsky_sparta_opp": "Sparta soupeř po Křetínském",
    "kretinsky_slavia": "Slavia po Křetínském",
    "kretinsky_slavia_opp": "Slavia soupeř po Křetínském",
    "kretinsky": "po Křetínském v lize celkem",
    "hga[T.H]": "výhoda domácího hřiště",
    }
    coefs["variable"] = coefs["variable"].replace(var_names_mapper)
    coefs["variable"] = pd.Categorical(coefs["variable"], categories=list(reversed(var_names_mapper.values())))

    # Compute the breaks for the error bars - get the min and max values and round to the nearest 0.1
    breaks_min = coefs["error_min"].min()
    breaks_max = coefs["error_max"].max()
    breaks = list(np.round(np.arange(np.floor(breaks_min * 10) / 10, np.ceil(breaks_max * 10) / 10 + 0.1, 0.1), 2))

    # Set custom theme
    custom_theme = p9.theme(
        axis_text_x=p9.element_text(color="black", size=12),
        axis_text_y=p9.element_text(color="black", size=12),
        axis_title=p9.element_text(color="black", size=12),
        plot_title=p9.element_text(color="black", size=14),
        axis_line=p9.element_line(color='black'),
        plot_background=p9.element_rect(fill='white'),
        panel_background=p9.element_rect(fill='white'),
        panel_grid_major_y=p9.element_blank(),
        panel_grid_minor_y=p9.element_blank()
    )
    # Plot the coefficient estimates with error bars
    coefs_plot = (
        p9.ggplot(coefs, p9.aes(x='variable', y='coefficient')) +
        p9.geom_hline(yintercept=0, linetype='dashed', color='grey') +
        p9.geom_errorbar(
            p9.aes(ymin='error_min', ymax='error_max'),
            width=0.2,
            color='grey',
            size=1.2,
            alpha=0.7
        ) +
        p9.geom_point(
            fill='blue',
            size=5,
            shape='o',
            stroke=0.5,
            color='black'
        ) +
        p9.coord_flip() +
        p9.scale_y_continuous(expand=(0.02, 0.02),
                            breaks=breaks,
                            labels=breaks) +
        p9.theme_minimal() +
        custom_theme +
        p9.labs(title= title, x='Variable', y='Coefficient') +
        p9.theme(
            #axis_title=p9.element_text(size=12),
            #axis_text=p9.element_text(size=10),
            #legend_title=p9.element_text(size=12),
            legend_text=p9.element_text(size=10)
        )
    )

    coefs_plot.save(file_name, width=16, height=9)
    # Display the coefficient plot
    print(coefs_plot)
    return coefs_plot
