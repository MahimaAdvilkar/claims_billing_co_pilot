class DecisionAgent:
    """
    Final decision-making agent.
    Aggregates insights from all other agents.
    Generates actionable alerts and decisions.
    """
    
    def run(self, state: dict) -> dict:
        decisions = []
        
        analysis = state.get("analysis", {})
        anomalies = state.get("anomalies", [])
        denial_analysis = state.get("denial_analysis", {})
        trend = state.get("trend_analysis", {})
        client_profit = state.get("client_profitability", [])
        
        try:
            # 1. VARIANCE ALERT
            if analysis.get("total_variance", 0) > 0:
                variance_pct = (analysis.get("total_variance", 0) / analysis.get("total_billed", 1)) * 100
                if variance_pct > 20:
                    decisions.append({
                        "type": "anomaly",
                        "message": f"⚠️ HIGH VARIANCE: {variance_pct:.1f}% of billed amount outstanding (${analysis.get('total_variance', 0):,.0f}). Recommend immediate payer reconciliation."
                    })
                elif variance_pct > 10:
                    decisions.append({
                        "type": "warning",
                        "message": f"📊 Moderate variance: {variance_pct:.1f}% outstanding. Monitor collections closely."
                    })
            
            # 2. COLLECTION RATE ALERT
            collection_rate = analysis.get("collection_rate", 0)
            if collection_rate < 80:
                decisions.append({
                    "type": "anomaly",
                    "message": f"❌ LOW COLLECTION RATE: Only {collection_rate:.1f}% collected. Below 80% threshold."
                })
            elif collection_rate < 90:
                decisions.append({
                    "type": "warning",
                    "message": f"⚠️ Collection rate at {collection_rate:.1f}%. Aim for 95%+ for optimal performance."
                })
            
            # 3. DENIAL RATE ALERT
            denial_rate = denial_analysis.get("denial_rate", 0)
            if denial_rate > 15:
                decisions.append({
                    "type": "anomaly",
                    "message": f"🚫 HIGH DENIAL RATE: {denial_rate:.1f}% of claims denied ({denial_analysis.get('total_denied', 0)} claims). Review top denial reasons."
                })
            elif denial_rate > 10:
                decisions.append({
                    "type": "warning",
                    "message": f"📋 Denial rate at {denial_rate:.1f}%. Investigate patterns with payers."
                })
            
            # 4. GROWTH TREND ALERT
            year_growth = trend.get("year_growth_pct", 0)
            if year_growth < 0:
                decisions.append({
                    "type": "anomaly",
                    "message": f"📉 NEGATIVE GROWTH: Year-over-year decline of {abs(year_growth):.1f}%. Investigate causes."
                })
            elif year_growth > 10:
                decisions.append({
                    "type": "alert",
                    "message": f"📈 Strong growth: {year_growth:.1f}% increase. Ensure infrastructure scales with demand."
                })
            
            # 5. RISKY CLIENTS ALERT
            if client_profit:
                risky_clients = [c for c in client_profit if c.get('Risk_Score', 0) > 60]
                if risky_clients:
                    high_risk = risky_clients[0]
                    decisions.append({
                        "type": "warning",
                        "message": f"⚠️ HIGH-RISK CLIENT: {high_risk.get('Client')} has risk score {high_risk.get('Risk_Score'):.0f}/100. Monitor closely."
                    })
            
            # 6. ANOMALIES SUMMARY
            if anomalies:
                high_variance_count = len([a for a in anomalies if a['type'] == 'high_variance'])
                high_denial_count = len([a for a in anomalies if a['type'] == 'high_denial_rate'])
                slow_payment_count = len([a for a in anomalies if a['type'] == 'slow_payment'])
                
                if high_variance_count > 0:
                    decisions.append({
                        "type": "warning",
                        "message": f"📊 {high_variance_count} client(s) with high variance (>15%). Review payment status."
                    })
                
                if high_denial_count > 0:
                    decisions.append({
                        "type": "warning",
                        "message": f"🚫 {high_denial_count} payer(s) with high denial rate (>12%). Investigate denial reasons."
                    })
                
                if slow_payment_count > 0:
                    decisions.append({
                        "type": "alert",
                        "message": f"⏱️ {slow_payment_count} month(s) with slow payments (>25 days). Contact payers for faster processing."
                    })
            else:
                decisions.append({
                    "type": "alert",
                    "message": "✅ No major anomalies detected. Billing operations are healthy."
                })
            
            print(f"✅ Decision analysis complete: {len(decisions)} alerts generated")
            return {"decisions": decisions}
            
        except Exception as e:
            print(f"❌ DecisionAgent Error: {str(e)}")
            return {"decisions": [{
                "type": "alert",
                "message": f"Decision analysis failed: {str(e)}"
            }]}