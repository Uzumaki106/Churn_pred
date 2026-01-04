import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import time

# Import the hyper-personalization engine from curn_pred.py
try:
    from churn_pred import HyperPersonalizationEngine, OpenRouterConfig, AIAgent
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    st.error("⚠️ Could not import HyperPersonalizationEngine. Make sure curn_pred.py is in the same directory.")

# Page config
st.set_page_config(
    page_title="AI Hyper-Personalization Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .customer-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ai_strategies' not in st.session_state:
    st.session_state.ai_strategies = {}
if 'api_key_1' not in st.session_state:
    st.session_state.api_key_1 = None
if 'api_key_2' not in st.session_state:
    st.session_state.api_key_2 = None
if 'api_key_3' not in st.session_state:
    st.session_state.api_key_3 = None

# Sidebar
st.sidebar.markdown("# 🤖 AI Platform")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "👥 Customer Intelligence", "📊 Analytics", "🎯 Personalization", "⚙️ Settings"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
if st.session_state.engine:
    st.sidebar.success("✅ Engine Running")
    st.sidebar.info(f"📊 {len(st.session_state.engine.customer_profiles)} Profiles")
else:
    st.sidebar.warning("⚠️ Engine Not Initialized")

# Main header
st.markdown('<h1 class="main-header">🤖 AI-Driven Hyper-Personalization & Churn Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown("---")

# ==============================================================================
# HELPER FUNCTION: Generate AI Strategy with Retry Logic
# ==============================================================================

def generate_ai_strategy_with_retry(engine, customer_id, profile, max_retries=3):
    """
    Generate AI strategy for a specific customer with retry logic
    Similar to the retry logic in curn_pred.py
    """
    for attempt in range(max_retries):
        try:
            with st.spinner(f"🤖 Attempt {attempt + 1}/{max_retries}: Generating AI strategy..."):
                # Use the engine's AI method
                strategy = engine.analyze_customer_with_ai(customer_id, profile)
                
                if strategy and 'intent' in strategy:
                    st.success(f"✅ AI strategy generated successfully on attempt {attempt + 1}!")
                    return strategy
                else:
                    st.warning(f"⚠️ Attempt {attempt + 1} returned incomplete data")
                    
        except Exception as e:
            st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
            
        if attempt < max_retries - 1:
            time.sleep(1)  # Wait before retry
    
    # If all retries fail, generate rule-based strategy
    st.info("📊 All AI attempts failed. Generating rule-based strategy as fallback...")
    return engine.generate_rule_based_strategy(profile)

# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================

if page == "🏠 Dashboard":
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("Real-time individual-centric customer intelligence")
    
    # File upload and initialization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type=['csv'])
        
    with col2:
        use_ai = st.checkbox("Enable AI Agents", value=True, help="Enable AI-powered analysis (requires API key)")
    
    if uploaded_file is not None:
        if st.button("🚀 Run Analysis", key="run_analysis"):
            with st.spinner("Initializing AI Hyper-Personalization Engine..."):
                # Save uploaded file temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Collect API keys from session state
                    api_keys = [k for k in [st.session_state.api_key_1, st.session_state.api_key_2, st.session_state.api_key_3] if k]
                    
                    # Initialize engine
                    st.session_state.engine = HyperPersonalizationEngine(
                        data_path=tmp_path,
                        use_ai_agents=use_ai,
                        api_keys=api_keys if api_keys else None
                    )
                    
                    # Run pipeline (load data and build profiles)
                    with st.spinner("Building customer profiles and predictions..."):
                        st.session_state.engine.load_and_prepare_data()
                        st.session_state.engine.build_ml_foundation()
                        st.session_state.engine.enrich_profiles_with_predictions()
                        
                        # Create results structure
                        results = {
                            'total_profiles': len(st.session_state.engine.customer_profiles),
                            'impact': {
                                'total_customers': len(st.session_state.engine.customer_profiles),
                                'high_risk': sum(1 for p in st.session_state.engine.customer_profiles.values() if p['churn_risk'] == 'High'),
                                'clv_at_risk': sum(p['lifetime_value'] for p in st.session_state.engine.customer_profiles.values() if p['churn_risk'] in ['High', 'Medium']),
                                'potential_savings': sum(p['lifetime_value'] for p in st.session_state.engine.customer_profiles.values() if p['churn_risk'] in ['High', 'Medium']) * 0.5
                            }
                        }
                        
                        st.session_state.results = results
                        st.session_state.analysis_complete = True
                    
                    st.success("✅ Analysis Complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        impact = results['impact']
        
        st.markdown("---")
        st.markdown("### 📈 Key Metrics")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 Total Customers</h3>
                <h2>{results['total_profiles']:,}</h2>
                <p>Individual Profiles</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ High Risk</h3>
                <h2>{impact['high_risk']}</h2>
                <p>Immediate Action Needed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 CLV at Risk</h3>
                <h2>${impact['clv_at_risk']:,.0f}</h2>
                <p>Potential Revenue Loss</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💵 Potential Savings</h3>
                <h2>${impact['potential_savings']:,.0f}</h2>
                <p>With AI Intervention</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Risk Distribution")
            
            # Count risk levels
            risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
            for profile in st.session_state.engine.customer_profiles.values():
                risk_counts[profile['churn_risk']] += 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                marker_colors=['#dc3545', '#ffc107', '#28a745']
            )])
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Top 10 Micro-Segments")
            
            # Count segments
            segment_counts = {}
            for profile in st.session_state.engine.customer_profiles.values():
                seg = profile['segment']
                segment_counts[seg] = segment_counts.get(seg, 0) + 1
            
            # Top 10 segments
            top_segments = sorted(segment_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            fig = go.Figure(data=[go.Bar(
                x=[f"Seg {s[0]}" for s in top_segments],
                y=[s[1] for s in top_segments],
                marker_color='#667eea'
            )])
            fig.update_layout(height=350, xaxis_title="Segment", yaxis_title="Customers")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top insights
        st.markdown("### 💡 AI-Generated Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("🎯 **Intent Prediction**\nAI can analyze individual customer intents on-demand for personalized strategies")
        
        with col2:
            st.success("📈 **Propensity Scores**\nPersonalized upsell opportunities identified for each customer")
        
        with col3:
            st.warning("🛡️ **Churn Prevention**\nProactive retention strategies can be generated per individual with AI")

# ==============================================================================
# PAGE 2: CUSTOMER INTELLIGENCE (MODIFIED WITH AI BUTTON)
# ==============================================================================

elif page == "👥 Customer Intelligence":
    st.markdown("## 👥 Individual Customer Intelligence")
    
    if not st.session_state.engine:
        st.warning("⚠️ Please run analysis from Dashboard first")
    else:
        engine = st.session_state.engine
        
        # Search/filter
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_id = st.text_input("🔍 Search Customer ID", placeholder="Enter customer ID...")
        
        with col2:
            risk_filter = st.selectbox("Filter by Risk", ["All", "High", "Medium", "Low"])
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Churn Risk", "CLV", "Tenure"])
        
        # Get filtered customers
        customers = list(engine.customer_profiles.items())
        
        if risk_filter != "All":
            customers = [(cid, p) for cid, p in customers if p['churn_risk'] == risk_filter]
        
        if search_id:
            customers = [(cid, p) for cid, p in customers if search_id.lower() in str(cid).lower()]
        
        # Sort
        if sort_by == "Churn Risk":
            customers = sorted(customers, key=lambda x: x[1]['churn_probability'], reverse=True)
        elif sort_by == "CLV":
            customers = sorted(customers, key=lambda x: x[1]['lifetime_value'], reverse=True)
        
        st.markdown(f"### Showing {len(customers)} customers")
        
        # Pagination
        items_per_page = 10
        total_pages = (len(customers) - 1) // items_per_page + 1 if len(customers) > 0 else 1
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Display customers
        for customer_id, profile in customers[start_idx:end_idx]:
            risk_class = f"risk-{profile['churn_risk'].lower()}"
            
            with st.expander(f"**{customer_id}** - {profile['churn_risk']} Risk ({profile['churn_probability']:.1%})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📊 Profile")
                    st.write(f"**CLV:** ${profile['lifetime_value']:,.2f}")
                    st.write(f"**Tenure:** {profile['features'].get('tenure', 'N/A')} months")
                    st.write(f"**Monthly:** ${profile['features'].get('MonthlyCharges', 'N/A')}")
                    st.write(f"**Segment:** #{profile['segment']}")
                
                with col2:
                    st.markdown("#### 🎯 Propensity Scores")
                    for key, score in profile['propensity_scores'].items():
                        st.progress(score, text=f"{key}: {score:.1%}")
                
                with col3:
                    st.markdown("#### 💡 Next Best Actions")
                    
                    # Check if AI strategy exists
                    if customer_id in st.session_state.ai_strategies:
                        actions = st.session_state.ai_strategies[customer_id].get('next_actions', ['No actions available'])
                        for action in actions[:3]:
                            st.write(f"• {action}")
                    else:
                        st.info("Click button below to generate AI-powered action plan")
                
                # AI Strategy Generation Button
                st.markdown("---")
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
                
                with col_btn1:
                    if st.button(f"🤖 Generate AI Actions", key=f"ai_btn_{customer_id}"):
                        # Generate strategy with retry logic
                        strategy = generate_ai_strategy_with_retry(engine, customer_id, profile)
                        
                        # Store in session state
                        st.session_state.ai_strategies[customer_id] = strategy
                        st.rerun()
                
                # Display AI strategy if generated
                if customer_id in st.session_state.ai_strategies:
                    with col_btn2:
                        if st.button(f"🗑️ Clear AI Data", key=f"clear_{customer_id}"):
                            del st.session_state.ai_strategies[customer_id]
                            st.rerun()
                    
                    st.markdown("---")
                    st.markdown("#### 🤖 AI-Generated Strategy")
                    
                    strategy = st.session_state.ai_strategies[customer_id]
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**🎯 Customer Intent**")
                        st.info(strategy.get('intent', 'Unknown'))
                        
                        st.markdown("**📊 Key Insights**")
                        insights = strategy.get('insights', ['No insights available'])
                        for insight in insights[:3]:
                            st.write(f"• {insight}")
                    
                    with col_b:
                        st.markdown("**✅ Recommended Actions**")
                        actions = strategy.get('next_actions', ['No actions available'])
                        for idx, action in enumerate(actions[:5], 1):
                            st.write(f"{idx}. {action}")
                    
                    # Show if AI-powered or rule-based
                    if strategy.get('ai_powered', False):
                        st.success("🤖 Generated with AI")
                    else:
                        st.warning("📊 Rule-based strategy (AI unavailable)")
                    
                    # Show full strategy as JSON
                    with st.expander("📋 View Full Strategy (JSON)"):
                        st.json(strategy)

# ==============================================================================
# PAGE 3: ANALYTICS
# ==============================================================================

elif page == "📊 Analytics":
    st.markdown("## 📊 Advanced Analytics")
    
    if not st.session_state.engine:
        st.warning("⚠️ Please run analysis from Dashboard first")
    else:
        engine = st.session_state.engine
        
        tab1, tab2, tab3 = st.tabs(["📈 Churn Analysis", "💰 Revenue Impact", "🎯 Propensity Analysis"])
        
        with tab1:
            st.markdown("### Churn Risk Analysis")
            
            # Create dataframe
            data = []
            for cid, profile in engine.customer_profiles.items():
                data.append({
                    'customer_id': cid,
                    'churn_prob': profile['churn_probability'],
                    'churn_risk': profile['churn_risk'],
                    'clv': profile['lifetime_value'],
                    'tenure': profile['features'].get('tenure', 0),
                    'monthly': profile['features'].get('MonthlyCharges', 0),
                    'segment': profile['segment']
                })
            
            df = pd.DataFrame(data)
            
            # Scatter plot
            fig = px.scatter(
                df,
                x='tenure',
                y='churn_prob',
                size='clv',
                color='churn_risk',
                hover_data=['customer_id', 'monthly'],
                title="Churn Probability vs Tenure (bubble size = CLV)",
                color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='churn_prob', nbins=50, title="Churn Probability Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, x='churn_risk', y='clv', title="CLV by Risk Level")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Revenue Impact Analysis")
            
            # Revenue at risk by segment
            segment_risk = df.groupby('segment').agg({
                'clv': 'sum',
                'churn_prob': 'mean'
            }).reset_index()
            segment_risk['revenue_at_risk'] = segment_risk['clv'] * segment_risk['churn_prob']
            segment_risk = segment_risk.sort_values('revenue_at_risk', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(name='Total CLV', x=segment_risk['segment'], y=segment_risk['clv']),
                go.Bar(name='Revenue at Risk', x=segment_risk['segment'], y=segment_risk['revenue_at_risk'])
            ])
            fig.update_layout(barmode='group', title='Top 10 Segments by Revenue at Risk')
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_clv = df['clv'].sum()
                st.metric("Total Customer Value", f"${total_clv:,.0f}")
            
            with col2:
                high_risk_clv = df[df['churn_risk'] == 'High']['clv'].sum()
                st.metric("High Risk CLV", f"${high_risk_clv:,.0f}", delta=f"-{high_risk_clv/total_clv*100:.1f}%")
            
            with col3:
                avg_clv = df['clv'].mean()
                st.metric("Average CLV", f"${avg_clv:,.0f}")
        
        with tab3:
            st.markdown("### Propensity Analysis")
            
            # Aggregate propensity scores
            propensity_data = []
            for profile in engine.customer_profiles.values():
                for key, score in profile['propensity_scores'].items():
                    propensity_data.append({
                        'product': key,
                        'score': score,
                        'risk': profile['churn_risk']
                    })
            
            prop_df = pd.DataFrame(propensity_data)
            
            # Average propensity by product
            fig = px.box(prop_df, x='product', y='score', color='risk',
                        title='Propensity Scores by Product and Risk Level')
            st.plotly_chart(fig, use_container_width=True)
            
            # Opportunity matrix
            st.markdown("### 🎯 Opportunity Matrix")
            
            high_prop = prop_df[prop_df['score'] > 0.7].groupby(['product', 'risk']).size().reset_index(name='count')
            
            fig = px.bar(high_prop, x='product', y='count', color='risk',
                        title='High Propensity Customers (>70%) by Product',
                        color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'})
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 4: PERSONALIZATION
# ==============================================================================

elif page == "🎯 Personalization":
    st.markdown("## 🎯 Personalization Campaigns")
    
    if not st.session_state.engine:
        st.warning("⚠️ Please run analysis from Dashboard first")
    else:
        engine = st.session_state.engine
        
        st.markdown("### Create Targeted Campaign")
        
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_type = st.selectbox(
                "Campaign Type",
                ["Retention (High Risk)", "Upsell (Low Risk)", "Cross-sell", "Win-back"]
            )
            
            target_segment = st.multiselect(
                "Target Segments",
                options=list(range(20)),
                default=[0, 1, 2]
            )
        
        with col2:
            risk_target = st.multiselect(
                "Risk Levels",
                ["High", "Medium", "Low"],
                default=["High"]
            )
            
            min_clv = st.number_input("Minimum CLV", value=500.0)
        
        if st.button("🎯 Generate Campaign"):
            # Filter customers
            target_customers = []
            for cid, profile in engine.customer_profiles.items():
                if (profile['churn_risk'] in risk_target and 
                    profile['segment'] in target_segment and
                    profile['lifetime_value'] >= min_clv):
                    target_customers.append((cid, profile))
            
            st.success(f"✅ Found {len(target_customers)} target customers")
            
            # Display sample
            st.markdown("### 📋 Sample Campaign Recipients")
            
            for cid, profile in target_customers[:5]:
                # Get next action
                if cid in st.session_state.ai_strategies:
                    actions = st.session_state.ai_strategies[cid].get('next_actions', [])
                    next_action = actions[0] if actions else "No action available"
                else:
                    next_action = "Generate AI action plan in Customer Intelligence"
                
                st.markdown(f"""
                <div class="customer-card">
                    <h4>{cid}</h4>
                    <p><strong>Risk:</strong> <span class="risk-{profile['churn_risk'].lower()}">{profile['churn_risk']}</span> | 
                    <strong>CLV:</strong> ${profile['lifetime_value']:,.0f} | 
                    <strong>Segment:</strong> #{profile['segment']}</p>
                    <p><strong>Recommended Action:</strong> {next_action}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Campaign summary
            total_clv = sum(p['lifetime_value'] for _, p in target_customers)
            avg_churn = np.mean([p['churn_probability'] for _, p in target_customers])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Customers", len(target_customers))
            with col2:
                st.metric("Total CLV", f"${total_clv:,.0f}")
            with col3:
                st.metric("Avg Churn Risk", f"{avg_churn:.1%}")

# ==============================================================================
# PAGE 5: SETTINGS (API KEYS MOVED HERE)
# ==============================================================================

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ System Settings")
    
    st.markdown("### 🤖 AI Configuration")
    
    st.info("""
    **OpenRouter API Keys Configuration**
    
    The system supports up to 3 OpenRouter API keys for automatic rotation on rate limits or errors.
    Get your free API keys from: **https://openrouter.ai/keys**
    
    Configure your API keys below. Changes will be saved automatically.
    """)
    
    # API Key inputs with session state
    api_key_1 = st.text_input(
        "API Key 1 (Primary)", 
        type="password",
        value=st.session_state.api_key_1,
        help="Primary API key for AI analysis"
    )
    
    api_key_2 = st.text_input(
        "API Key 2 (Backup)", 
        type="password",
        value=st.session_state.api_key_2,
        help="Backup API key (optional)"
    )
    
    api_key_3 = st.text_input(
        "API Key 3 (Backup)", 
        type="password",
        value=st.session_state.api_key_3,
        help="Second backup API key (optional)"
    )
    
    if st.button("💾 Save API Keys"):
        st.session_state.api_key_1 = api_key_1
        st.session_state.api_key_2 = api_key_2
        st.session_state.api_key_3 = api_key_3
        st.success("✅ API Keys saved! They will be used in the next analysis run.")
    
    st.markdown("---")
    
    # Model information
    model_info = st.expander("📋 Available AI Models (Automatic Fallback)", expanded=False)
    with model_info:
        st.markdown("""
        The system uses the following free models with automatic fallback:
        
        1. **meta-llama/llama-3.2-3b-instruct:free** - Primary (Fast and efficient)
        2. **google/gemini-flash-1.5:free** - Fallback 1 (Google's model)
        3. **nousresearch/hermes-3-llama-3.1-405b:free** - Fallback 2 (Powerful model)
        
        Models are automatically rotated if one fails or hits rate limits.
        Each customer AI analysis includes 3 retry attempts before falling back to rule-based strategy.
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Data Configuration")
    
    segment_count = st.slider("Number of Micro-Segments", 5, 50, 20)
    st.info(f"System will create {segment_count} customer micro-segments for detailed analysis")
    
    st.markdown("---")
    st.markdown("### 🔄 System Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear AI Strategies Cache"):
            st.session_state.ai_strategies = {}
            st.success(f"✅ Cleared {len(st.session_state.ai_strategies)} AI strategies from cache")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Engine"):
            st.session_state.engine = None
            st.session_state.analysis_complete = False
            st.session_state.results = None
            st.session_state.ai_strategies = {}
            st.success("✅ Engine reset. Please re-run analysis from Dashboard.")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Customer Profiles"):
            if st.session_state.engine:
                # Create export data
                export_data = []
                for cid, profile in st.session_state.engine.customer_profiles.items():
                    export_data.append({
                        'Customer ID': cid,
                        'Churn Risk': profile['churn_risk'],
                        'Churn Probability': profile['churn_probability'],
                        'CLV': profile['lifetime_value'],
                        'Segment': profile['segment'],
                        'Tenure': profile['features'].get('tenure', 'N/A'),
                        'Monthly Charges': profile['features'].get('MonthlyCharges', 'N/A')
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"customer_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No data to export. Please run analysis first.")
    
    with col2:
        if st.button("📥 Export AI Strategies"):
            if st.session_state.ai_strategies:
                # Convert to JSON
                json_data = json.dumps(st.session_state.ai_strategies, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"ai_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("⚠️ No AI strategies to export. Generate some first in Customer Intelligence page.")
    
    st.markdown("---")
    st.markdown("### 📊 Current System Stats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        profiles_count = len(st.session_state.engine.customer_profiles) if st.session_state.engine else 0
        st.metric("Customer Profiles", profiles_count)
    
    with col2:
        ai_count = len(st.session_state.ai_strategies)
        st.metric("AI Strategies Generated", ai_count)
    
    with col3:
        api_keys_configured = sum(1 for k in [st.session_state.api_key_1, st.session_state.api_key_2, st.session_state.api_key_3] if k)
        st.metric("API Keys Configured", api_keys_configured)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    **AI-Driven Hyper-Personalization Platform**
    
    Version: 2.0.0
    
    This platform uses AI agents to create individual customer intelligence profiles
    and generate personalized strategies on-demand with automatic retry logic.
    
    **Key Features:**
    - Individual customer profiling (not personas)
    - On-demand AI-powered action generation
    - Automatic retry logic (3 attempts per customer)
    - API key rotation for reliability
    - Real-time propensity scoring
    - Proactive churn prevention
    - Personalized campaign generation
    
    **What's New in v2.0:**
    - Changed from "Analyze top N" to on-demand AI generation per customer
    - Added retry logic for robust AI response generation
    - API keys configuration moved to Settings page
    - Improved customer intelligence interface with AI action buttons
    - Added export functionality for profiles and strategies
    - Enhanced dashboard with all original visualizations
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 <strong>AI-Driven Hyper-Personalization & Churn Intelligence Platform</strong></p>
    <p>Individual-Centric Intelligence • On-Demand AI Actions • Automatic Retry Logic</p>
</div>
""", unsafe_allow_html=True)
