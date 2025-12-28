import pandas as pd

class DenialReasonAgent:
    """
    Analyze denial patterns:
    - Top denial reasons
    - Denial count by reason
    - Denial amount by reason
    - Denial breakdown by payer
    """
    
    def run(self, state: dict) -> dict:
        df = state.get("master_df")
        
        if df is None or len(df) == 0:
            print("⚠️ DenialReasonAgent: No data available")
            return {"denial_analysis": {}}
        
        try:
            # Filter denied claims
            denied_df = df[df['Status'] == 'Denied'].copy()
            
            if len(denied_df) == 0:
                print("✅ No denied claims found")
                return {
                    "denial_analysis": {
                        "total_denied": 0,
                        "total_denied_amount": 0.0,
                        "denial_rate": 0.0,
                        "reasons": [],
                        "by_payer": []
                    }
                }
            
            # 1. TOP DENIAL REASONS
            denial_reasons = denied_df.groupby('Denial_Reason').agg({
                'Claim_ID': 'count',
                'Amount_Billed': 'sum'
            }).reset_index()
            
            denial_reasons.columns = ['Reason', 'Count', 'Amount']
            denial_reasons['Amount'] = denial_reasons['Amount'].round(2)
            denial_reasons = denial_reasons.sort_values('Count', ascending=False)
            
            # 2. DENIAL BY PAYER
            denial_by_payer = denied_df.groupby('Payer').agg({
                'Claim_ID': 'count',
                'Amount_Billed': 'sum'
            }).reset_index()
            
            denial_by_payer.columns = ['Payer', 'Denial_Count', 'Denial_Amount']
            denial_by_payer['Denial_Amount'] = denial_by_payer['Denial_Amount'].round(2)
            denial_by_payer = denial_by_payer.sort_values('Denial_Count', ascending=False)
            
            # 3. OVERALL METRICS
            total_denied = len(denied_df)
            total_denied_amount = denied_df['Amount_Billed'].sum()
            denial_rate = (total_denied / len(df) * 100) if len(df) > 0 else 0
            
            print(f"✅ Denial analysis complete: {total_denied} denied claims ({denial_rate:.1f}%)")
            return {
                "denial_analysis": {
                    "total_denied": int(total_denied),
                    "total_denied_amount": float(total_denied_amount),
                    "denial_rate": float(round(denial_rate, 2)),
                    "reasons": denial_reasons.to_dict('records'),
                    "by_payer": denial_by_payer.to_dict('records')
                }
            }
            
        except Exception as e:
            print(f"❌ DenialReasonAgent Error: {str(e)}")
            return {"denial_analysis": {}}