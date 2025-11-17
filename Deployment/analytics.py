import io, base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return b64

def generate_eda_report_html(df: pd.DataFrame, categorical_cols: list, numeric_cols: list) -> Tuple[str, bytes]:
    parts = [
        "<h1>FraudDetection BFSI — EDA Report</h1>",
        f"<p>Generated: {pd.Timestamp.utcnow().isoformat()} UTC</p>",
        "<h2>Basic summary</h2>",
        f"<pre>{df.describe(include='all').to_html()}</pre>"
    ]
    # --- analytics.py visuals ---
    if 'Risk_Category' in df.columns:
        counts = df['Risk_Category'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Risk Category Distribution')
        parts.append("<h2>Risk Category Distribution</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    for c in categorical_cols:
        if c in df.columns:
            vc = df[c].value_counts().head(20)
            fig, ax = plt.subplots(figsize=(5, 2.5))
            vc.plot.bar(ax=ax)
            ax.set_title(f'{c} (top categories)')
            ax.set_ylabel('count')
            parts.append(f"<h3>{c}</h3>")
            parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    parts.append("<h2>Numerical distributions</h2>")
    for n in numeric_cols:
        if n in df.columns:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            df[n].dropna().astype(float).hist(bins=40, ax=ax)
            ax.set_title(n)
            parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if not num_df.empty and num_df.shape[1] > 1:
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(cax)
        ax.set_title('Correlation matrix')
        parts.append("<h2>Correlation matrix</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')

    # --- MODULE 1 additional visuals ---
    # Fraud Distribution Pie (different column: isFraud)
    if 'isFraud' in df.columns:
        fraud_counts = df['isFraud'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
               colors=['lightgreen', 'red'])
        ax.set_title('Fraud Distribution')
        parts.append("<h2>Fraud Distribution</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    # Transaction Amount vs Fraud Status (Boxplot)
    if 'isFraud' in df.columns and 'Transaction_Amount' in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(x='isFraud', y='Transaction_Amount', data=df, ax=ax)
        ax.set_title('Transaction Amount by Fraud Status')
        ax.set_yscale('log')
        parts.append("<h2>Transaction Amount by Fraud Status</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    # Fraud Rate by Hour
    if 'isFraud' in df.columns and 'Transaction_Hour' in df.columns:
        fraud_by_hour = df.groupby('Transaction_Hour')['isFraud'].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(fraud_by_hour.index, fraud_by_hour.values, marker='o')
        ax.set_title('Fraud Rate by Hour of Day')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Fraud Rate')
        parts.append("<h2>Fraud Rate by Hour of Day</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    # Top 10 Locations by Fraud Rate
    if 'Transaction_Location' in df.columns and 'isFraud' in df.columns:
        loc_fraud = df.groupby('Transaction_Location')['isFraud'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 3))
        loc_fraud.plot(kind='bar', color='orange', ax=ax)
        ax.set_title('Top 10 Locations by Fraud Rate')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        parts.append("<h2>Top 10 Locations by Fraud Rate</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    # Fraud Rate by Card Type
    if 'Card_Type' in df.columns and 'isFraud' in df.columns:
        card_fraud = df.groupby('Card_Type')['isFraud'].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        card_fraud.plot(kind='bar', color=['blue', 'orange'], ax=ax)
        ax.set_title('Fraud Rate by Card Type')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        parts.append("<h2>Fraud Rate by Card Type</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    # Fraud Rate by Authentication Method
    if 'Authentication_Method' in df.columns and 'isFraud' in df.columns:
        auth_fraud = df.groupby('Authentication_Method')['isFraud'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        auth_fraud.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Fraud Rate by Authentication Method')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        parts.append("<h2>Fraud Rate by Authentication Method</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')
    # Fraud Rate by Transaction Category
    if 'Transaction_Category' in df.columns and 'isFraud' in df.columns:
        cat_fraud = df.groupby('Transaction_Category')['isFraud'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        cat_fraud.plot(kind='bar', color='lightcoral', ax=ax)
        ax.set_title('Fraud Rate by Transaction Category')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        parts.append("<h2>Fraud Rate by Transaction Category</h2>")
        parts.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />')

    html = "<html><body>" + "\n".join(parts) + "</body></html>"
    return html, html.encode('utf-8')

def display_eda_inline(st, df: pd.DataFrame, categorical_cols: list, numeric_cols: list):
    st.markdown("## EDA — Overview")
    # ---- analytics.py visuals (Risk Category, Categorical, Numeric, Correlation) ----
    if 'Risk_Category' in df.columns:
        st.markdown("### Risk Category Distribution")
        counts = df['Risk_Category'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)
    st.markdown("### Categorical distributions (sample)")
    shown = 0
    for c in categorical_cols:
        if c in df.columns and shown < 3:
            st.markdown(f"**{c}**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            df[c].value_counts().head(20).plot.bar(ax=ax)
            st.pyplot(fig)
            shown += 1
    st.markdown("### Numerical distributions (sample)")
    shown = 0
    for n in numeric_cols:
        if n in df.columns and shown < 3:
            st.markdown(f"**{n}**")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            df[n].dropna().astype(float).hist(bins=40, ax=ax)
            st.pyplot(fig)
            shown += 1
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    if not num_df.empty and num_df.shape[1] > 1:
        st.markdown("### Correlation heatmap")
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(cax)
        st.pyplot(fig)
    # ---- MODULE 1 visuals ----
    if 'isFraud' in df.columns:
        st.markdown("### Fraud Distribution")
        fraud_counts = df['isFraud'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%', colors=['lightgreen', 'red'])
        ax.set_title('Fraud Distribution')
        st.pyplot(fig)
    if 'isFraud' in df.columns and 'Transaction_Amount' in df.columns:
        st.markdown("### Transaction Amount by Fraud Status")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.boxplot(x='isFraud', y='Transaction_Amount', data=df, ax=ax)
        ax.set_title('Transaction Amount by Fraud Status')
        ax.set_yscale('log')
        st.pyplot(fig)
    if 'isFraud' in df.columns and 'Transaction_Hour' in df.columns:
        st.markdown("### Fraud Rate by Hour of Day")
        fraud_by_hour = df.groupby('Transaction_Hour')['isFraud'].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(fraud_by_hour.index, fraud_by_hour.values, marker='o')
        ax.set_title('Fraud Rate by Hour of Day')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Fraud Rate')
        st.pyplot(fig)
    if 'Transaction_Location' in df.columns and 'isFraud' in df.columns:
        st.markdown("### Top 10 Locations by Fraud Rate")
        loc_fraud = df.groupby('Transaction_Location')['isFraud'].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5, 3))
        loc_fraud.plot(kind='bar', color='orange', ax=ax)
        ax.set_title('Top 10 Locations by Fraud Rate')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    if 'Card_Type' in df.columns and 'isFraud' in df.columns:
        st.markdown("### Fraud Rate by Card Type")
        card_fraud = df.groupby('Card_Type')['isFraud'].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        card_fraud.plot(kind='bar', color=['blue', 'orange'], ax=ax)
        ax.set_title('Fraud Rate by Card Type')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)
    if 'Authentication_Method' in df.columns and 'isFraud' in df.columns:
        st.markdown("### Fraud Rate by Authentication Method")
        auth_fraud = df.groupby('Authentication_Method')['isFraud'].mean().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        auth_fraud.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title('Fraud Rate by Authentication Method')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
    if 'Transaction_Category' in df.columns and 'isFraud' in df.columns:
        st.markdown("### Fraud Rate by Transaction Category")
        cat_fraud = df.groupby('Transaction_Category')['isFraud'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        cat_fraud.plot(kind='bar', color='lightcoral', ax=ax)
        ax.set_title('Fraud Rate by Transaction Category')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
