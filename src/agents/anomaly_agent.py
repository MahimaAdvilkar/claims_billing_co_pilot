import pandas as pd

class AnomalyAgent:
    """
    Detect billing anomalies:
    1. High variance clients (>15% gap between billed and received)
    2. High denial rate payers (>12% denial rate)
    3. Slow payment months (>25 days average)
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        if df is None or len(df) == 0:
            print("⚠️ AnomalyAgent: No data available")
            return {"anomalies": []}
        
        try:
            anomalies = []
            
            # 1. HIGH VARIANCE CLIENTS (>15% gap)
            client_variance = df.groupby('Client_Name').agg({
                'Amount_Billed': 'sum',
                'Amount_Received': 'sum'
            }).reset_index()
            
            client_variance['Variance'] = client_variance['Amount_Billed'] - client_variance['Amount_Received']
            client_variance['Variance_Pct'] = (
                (client_variance['Variance'] / client_variance['Amount_Billed'] * 100)
                .fillna(0).round(2)
            )
            
            high_variance = client_variance[client_variance['Variance_Pct'] > 15]
            for _, row in high_variance.iterrows():
                anomalies.append({
                    "type": "high_variance",
                    "client": row['Client_Name'],
                    "variance": float(row['Variance']),
                    "variance_pct": float(row['Variance_Pct'])
                })
            
            # 2. HIGH DENIAL RATE PAYERS (>12%)
            denial_by_payer = df[df['Status'] == 'Denied'].groupby('Payer').size()
            total_by_payer = df.groupby('Payer').size()
            denial_rate = (denial_by_payer / total_by_payer * 100).round(2)
            
            high_denial = denial_rate[denial_rate > 12]
            for payer, rate in high_denial.items():
                anomalies.append({
                    "type": "high_denial_rate",
                    "payer": payer,
                    "denial_rate": float(rate),
                    "denied_count": int(denial_by_payer.get(payer, 0))
                })
            
            # 3. SLOW PAYMENT MONTHS (>25 days average)
            monthly_payment = df.groupby('Service_Month')['Days_To_Payment'].mean()
            slow_months = monthly_payment[monthly_payment > 25]
            
            for month, days in slow_months.items():
                anomalies.append({
                    "type": "slow_payment",
                    "month": month,
                    "avg_days": float(days)
                })
            
            if anomalies:
                print(f"⚠️ Anomalies detected: {len(anomalies)} issues found")
            else:
                print(f"✅ No anomalies detected")
            
            return {"anomalies": anomalies}
            
        except Exception as e:
            print(f"❌ AnomalyAgent Error: {str(e)}")
            return {"anomalies": []}