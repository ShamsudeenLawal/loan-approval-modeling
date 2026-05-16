# EDA HELPER FUNCTIONS
# NUMERICAL FEATURES (VS CATEGORICAL TARGET)

def plot_numerical_distributions(
    df,
    numerical_columns,
    target_column:str=None,
    plot_types="auto",   # 'hist', 'kde', 'box', 'violin', 'auto'
    bins=30,
    cols=3,
    figsize_per_plot=(5,4),
    palette="Set2",
    kde=True
):
    """
    Visualize distributions of numerical variables.

    Parameters:
    - df (pd.DataFrame)
    - numerical_columns (list): numerical features to plot
    - target_column (str, optional): if given, separate distributions by target
    - plot_types (str or dict): plot type per column ('hist', 'kde', 'box', 'violin', 'auto')
    - bins (int): number of bins for histograms
    - cols (int): number of columns per row
    - figsize_per_plot (tuple)
    - palette (str/list)
    - kde (bool): show KDE in hist plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if not isinstance(numerical_columns, (list, tuple)):
        raise ValueError("numerical_columns must be a list or tuple")

    missing_cols = [col for col in numerical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Normalize plot_types
    if plot_types == "auto":
        plot_types = {col: "hist" for col in numerical_columns}
    elif isinstance(plot_types, str):
        plot_types = {col: plot_types for col in numerical_columns}
    elif isinstance(plot_types, dict):
        plot_types = {col: plot_types.get(col, "hist") for col in numerical_columns}
    else:
        raise ValueError("plot_types must be 'auto', string, or dict")

    # Plot order
    num_plots = len(numerical_columns)
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig_width = cols * figsize_per_plot[0]
    fig_height = rows * figsize_per_plot[1]
    plt.figure(figsize=(fig_width, fig_height))

    for idx, col in enumerate(numerical_columns):
        ax = plt.subplot(rows, cols, idx + 1)
        ptype = plot_types.get(col, "hist")

        # Histogram / KDE
        if ptype == "hist":
            if target_column:
                sns.histplot(data=df, x=col, hue=target_column, kde=kde, bins=bins, palette=palette, ax=ax)
            else:
                sns.histplot(data=df, x=col, kde=kde, bins=bins, color=sns.color_palette(palette)[0], ax=ax)
            ax.set_title(f"{col} Histogram")

        # KDE only
        elif ptype == "kde":
            if target_column:
                sns.kdeplot(data=df, x=col, hue=target_column, fill=True, palette=palette, ax=ax)
            else:
                sns.kdeplot(data=df, x=col, fill=True, color=sns.color_palette(palette)[0], ax=ax)
            ax.set_title(f"{col} KDE")

        # Boxplot
        elif ptype == "box":
            if target_column:
                sns.boxplot(x=target_column, y=col, data=df, palette=palette, ax=ax)
                ax.set_title(f"{col} by {target_column} (Boxplot)")
            else:
                sns.boxplot(x=df[col], color=sns.color_palette(palette)[0], ax=ax)
                ax.set_title(f"{col} Boxplot")

        # Violin
        elif ptype == "violin":
            if target_column:
                sns.violinplot(x=target_column, y=col, data=df, palette=palette, ax=ax)
                ax.set_title(f"{col} by {target_column} (Violin)")
            else:
                sns.violinplot(x=df[col], color=sns.color_palette(palette)[0], ax=ax)
                ax.set_title(f"{col} Violin Plot")

        else:
            raise ValueError(f"Unsupported plot type '{ptype}' for column '{col}'")

        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


# CATEGORICAL DISTRIBUTIONS
def plot_categorical_distributions(
    df,
    categorical_columns,
    plot_type="bar",
    cols=3,
    figsize_per_plot=(5,4),
    palette="Set2",
    normalize=False,
    show_labels=True,
    xtick_rotation=0
):
    """
    Visualize distribution of categorical variables.

    Parameters:
    - df (pd.DataFrame)
    - categorical_columns (list)
    - plot_type (str): "bar" or "pie"
    - cols (int): number of columns per row
    - figsize_per_plot (tuple)
    - palette (str/list)
    - normalize (bool): show proportions instead of counts
    - show_labels (bool): annotate values on bars
    - xtick_rotation (int/float)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # -------- Validation --------
    if not isinstance(categorical_columns, (list, tuple)):
        raise ValueError("categorical_columns must be a list or tuple")

    missing_cols = [col for col in categorical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    num_plots = len(categorical_columns)
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig_width = cols * figsize_per_plot[0]
    fig_height = rows * figsize_per_plot[1]

    plt.figure(figsize=(fig_width, fig_height))

    # -------- Plotting --------
    for idx, col in enumerate(categorical_columns):
        ax = plt.subplot(rows, cols, idx + 1)

        value_counts = df[col].value_counts(normalize=normalize)
        values = value_counts.values
        labels = value_counts.index.astype(str)

        # ------------------ BAR ------------------
        if plot_type == "bar":
            sns.barplot(
                x=labels,
                y=values,
                palette=palette,
                ax=ax,
                hue=labels,
                legend=False,
            )

            ax.set_title(f"{col} Distribution")
            ax.tick_params(axis='x', rotation=xtick_rotation)

            if show_labels:
                for i, v in enumerate(values):
                    ax.text(i, v, f"{v:.2f}" if normalize else f"{int(v)}",
                            ha='center', va='bottom', fontsize=9)

        # ------------------ PIE ------------------
        elif plot_type == "pie":
            ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette(palette, len(values))
            )
            ax.set_title(f"{col} Distribution")

        else:
            raise ValueError("plot_type must be 'bar' or 'pie'")

        ax.set_xlabel("")
        ax.set_ylabel("Proportion" if normalize else "Count")

    plt.tight_layout()
    plt.show()


# CATEGORICAL TARGET VS CATEGORICAL FEATURES DISTRIBUTIONS
def plot_target_vs_categorical_features(
    df,
    categorical_columns,
    target_column=None,
    plot_types="bar",
    xtick_rotation=45,
    cols=3,
    figsize_per_plot=(5, 4),
    palette="Set2"
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # -------- Validation --------
    if not isinstance(categorical_columns, (list, tuple)):
        raise ValueError("categorical_columns must be a list or tuple")

    missing_cols = [col for col in categorical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # -------- Normalize plot_types --------
    if isinstance(plot_types, str):
        plot_types = {col: plot_types for col in categorical_columns}
    elif isinstance(plot_types, dict):
        plot_types = {col: plot_types.get(col, "bar") for col in categorical_columns}
    else:
        raise ValueError("plot_types must be a string or dict")

    # -------- Normalize xtick_rotation --------
    if isinstance(xtick_rotation, (int, float)):
        xtick_rotation = {col: xtick_rotation for col in categorical_columns}
    elif isinstance(xtick_rotation, dict):
        xtick_rotation = {col: xtick_rotation.get(col, 0) for col in categorical_columns}
    else:
        raise ValueError("xtick_rotation must be an int, float, or dict")

    # -------- Build plot order (TARGET FIRST) --------
    plot_columns = []

    if target_column:
        plot_columns.append(target_column)

    for col in categorical_columns:
        if col != target_column:
            plot_columns.append(col)

    # Ensure configs include target
    if target_column:
        plot_types[target_column] = plot_types.get(target_column, "bar")
        xtick_rotation[target_column] = xtick_rotation.get(target_column, 0)

    num_plots = len(plot_columns)
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig_width = cols * figsize_per_plot[0]
    fig_height = rows * figsize_per_plot[1]

    plt.figure(figsize=(fig_width, fig_height))

    for idx, col in enumerate(plot_columns):
        ax = plt.subplot(rows, cols, idx + 1)
        current_plot_type = plot_types.get(col, "bar")

        # ------------------ BAR ------------------
        if current_plot_type == "bar":
            if target_column and col != target_column:
                sns.countplot(
                    data=df,
                    x=col,
                    hue=target_column,
                    palette=palette,
                    ax=ax
                )
                ax.set_title(f"{col} vs {target_column}")
            else:
                sns.countplot(
                    data=df,
                    x=col,
                    hue=col,
                    palette=palette,
                    legend=False,
                    ax=ax
                )
                ax.set_title(f"Distribution of {col}")

            # Apply rotation
            rotation_angle = xtick_rotation.get(col, 0)
            ax.tick_params(axis='x', rotation=rotation_angle)

            # Count labels
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height)}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

        # ------------------ PIE ------------------
        elif current_plot_type == "pie":
            counts = df[col].value_counts()
            labels = counts.index.astype(str)

            ax.pie(
                counts,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette(palette, len(counts))
            )
            ax.set_title(f"{col} Distribution")

        else:
            raise ValueError(f"Unsupported plot type '{current_plot_type}' for column '{col}'")

        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.show()


# CATEGORICAL TARGET VS NUMERICAL FEATURES
def eda_categorical_vs_numerical(
    df,
    numerical_columns,
    label_column,
    plot_types=("box", "violin", "bar"),
    figsize_per_plot=(5,4),
    cols=3,
    palette="Set2",
    show_counts=True
):
    """
    Perform EDA for a categorical label against numerical features.

    Parameters:
    - df (pd.DataFrame)
    - numerical_columns (list): numerical features to analyze
    - label_column (str): categorical target column
    - plot_types (tuple): types of plots per feature: "box", "violin", "bar"
    - figsize_per_plot (tuple): width & height per subplot
    - cols (int): number of columns per row
    - palette (str/list): color palette for plots
    - show_counts (bool): if True, show mean value on top of bar plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    
    missing_cols = [col for col in numerical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Numerical columns not found: {missing_cols}")
    
    num_features = len(numerical_columns)
    rows = (num_features // cols) + (num_features % cols > 0)
    
    for plot_type in plot_types:
        fig_width = cols * figsize_per_plot[0]
        fig_height = rows * figsize_per_plot[1]
        plt.figure(figsize=(fig_width, fig_height))
        
        for idx, col in enumerate(numerical_columns):
            ax = plt.subplot(rows, cols, idx + 1)
            
            if plot_type == "box":
                sns.boxplot(x=label_column, y=col, data=df, palette=palette, ax=ax, hue=label_column, legend=False)
                ax.set_title(f"Boxplot of {col} by {label_column}")
                
            elif plot_type == "violin":
                sns.violinplot(x=label_column, y=col, data=df, palette=palette, ax=ax, hue=label_column, legend=False)
                ax.set_title(f"Violin plot of {col} by {label_column}")
                
            elif plot_type == "bar":
                means = df.groupby(label_column)[col].mean().reset_index()
                sns.barplot(x=label_column, y=col, data=means, palette=palette, ax=ax, hue=label_column, legend=False)
                ax.set_title(f"Mean {col} by {label_column}")
                
                if show_counts:
                    for p in ax.patches:
                        height = p.get_height()
                        ax.annotate(
                            f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )
            else:
                raise ValueError(f"Unsupported plot type '{plot_type}'")
            
        plt.tight_layout()
        plt.show()


# NUMERICAL VARIABLE VS NUMERICAL VARIABLE
def eda_numerical_vs_numerical(
    df,
    numerical_columns,
    label_column,
    plot_types=("scatter", "reg", "hex", "residual"),
    figsize_per_plot=(5,4),
    cols=2,
    palette="Set1",
    show_corr=True,
    corr_method="pearson",
    show_distribution=True,
    show_corr_matrix=True,
    corr_figsize=(10,8)
):
    """
    Perform EDA for numerical features vs a numerical label using pandas correlation.

    Includes:
    - Feature vs target plots
    - Distribution plots
    - Full correlation heatmap

    Parameters:
    - df (pd.DataFrame)
    - numerical_columns (list)
    - label_column (str)
    - plot_types (tuple)
    - figsize_per_plot (tuple)
    - cols (int)
    - palette (str/list)
    - show_corr (bool)
    - corr_method (str): 'pearson', 'spearman', 'kendall'
    - show_distribution (bool)
    - show_corr_matrix (bool)
    - corr_figsize (tuple)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # -------------------------
    # Validation
    # -------------------------
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    
    missing_cols = [col for col in numerical_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Numerical columns not found: {missing_cols}")

    if not pd.api.types.is_numeric_dtype(df[label_column]):
        raise ValueError(f"Label column '{label_column}' must be numerical")

    # -------------------------
    # Correlation Matrix (Pandas)
    # -------------------------
    df_numeric = df[numerical_columns + [label_column]].dropna()
    corr_matrix = df_numeric.corr(method=corr_method)

    # Extract feature vs target correlations
    corr_series = corr_matrix[label_column].drop(label_column)
    corr_df = corr_series.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    corr_df = corr_df.sort_values(by="Correlation", ascending=False)

    # -------------------------
    # FULL CORRELATION HEATMAP
    # -------------------------
    if show_corr_matrix:
        plt.figure(figsize=corr_figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            cbar=True
        )
        plt.title(f"Full Correlation Matrix ({corr_method.capitalize()})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # -------------------------
    # Feature vs Target Plots
    # -------------------------
    num_features = len(numerical_columns)
    rows = (num_features // cols) + (num_features % cols > 0)

    for plot_type in plot_types:
        fig_width = cols * figsize_per_plot[0]
        fig_height = rows * figsize_per_plot[1]
        plt.figure(figsize=(fig_width, fig_height))

        for idx, col in enumerate(numerical_columns):
            ax = plt.subplot(rows, cols, idx + 1)

            temp = df[[col, label_column]].dropna()
            corr = corr_df.loc[corr_df["Feature"] == col, "Correlation"].values[0]
            title_suffix = f"(corr={corr:.2f})" if show_corr else ""

            if plot_type == "scatter":
                sns.scatterplot(x=col, y=label_column, data=temp, ax=ax, alpha=0.6)
                ax.set_title(f"{col} vs {label_column} {title_suffix}")

            elif plot_type == "reg":
                sns.regplot(
                    x=col,
                    y=label_column,
                    data=temp,
                    ax=ax,
                    scatter_kws={"alpha": 0.5},
                    line_kws={"color": "red"}
                )
                ax.set_title(f"Regression: {col} vs {label_column} {title_suffix}")

            elif plot_type == "hex":
                ax.hexbin(temp[col], temp[label_column], gridsize=25, cmap="Blues")
                ax.set_title(f"Hexbin: {col} vs {label_column}")

            elif plot_type == "residual":
                sns.residplot(
                    x=col,
                    y=label_column,
                    data=temp,
                    lowess=True,
                    ax=ax,
                    scatter_kws={"alpha": 0.5}
                )
                ax.set_title(f"Residual: {col} vs {label_column}")

            else:
                raise ValueError(f"Unsupported plot type '{plot_type}'")

        plt.tight_layout()
        plt.show()

    # -------------------------
    # Distribution Plots
    # -------------------------
    if show_distribution:
        for col in numerical_columns:
            temp = df[[col]].dropna()

            fig, axes = plt.subplots(1, 2, figsize=(10,4))

            sns.histplot(temp[col], kde=True, ax=axes[0])
            axes[0].set_title(f"{col} Distribution")

            sns.boxplot(x=temp[col], ax=axes[1])
            axes[1].set_title(f"{col} Boxplot")

            plt.tight_layout()
            plt.show()

    # -------------------------
    # Output
    # -------------------------
    print("\nCorrelation Summary (Feature vs Target):")
    print(corr_df)

    return corr_df
