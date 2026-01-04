import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
import warnings
import os
import json
from datetime import datetime
import requests
import time
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# OPENROUTER CONFIGURATION WITH API KEY ROTATION
# ==============================================================================

class OpenRouterConfig:
    """Configuration for OpenRouter with API key rotation
    
    Supports up to 3 API keys with automatic rotation on rate limits or errors.
    Get free API keys from: https://openrouter.ai/keys
    """
    
    def __init__(self, api_keys=None):
        # Support for 3 API keys with rotation
        if api_keys is None:
            api_keys = [
                os.environ.get('OPENROUTER_API_KEY_1'),
                os.environ.get('OPENROUTER_API_KEY_2'),
                os.environ.get('OPENROUTER_API_KEY_3')
            ]
        
        # Filter out None values
        self.api_keys = [key for key in api_keys if key]
        
        if not self.api_keys:
            raise ValueError("At least one OpenRouter API key must be provided")
        
        self.current_key_index = 0
        self.api_base = "https://openrouter.ai/api/v1"
        
        # Using free OpenRouter models (fallback chain)
        # These models don't require privacy policy configuration
        self.models = [
            'meta-llama/llama-3.2-3b-instruct:free',  # Primary: Fast and free
            'google/gemini-flash-1.5:free',           # Fallback 1: Google's free model
            'nousresearch/hermes-3-llama-3.1-405b:free'  # Fallback 2: Powerful free model
        ]
        self.current_model_index = 0
        self.model_name = self.models[self.current_model_index]
        
        # Generation config
        self.temperature = 0.7
        self.max_tokens = 2048
        self.top_p = 0.95
        
        print(f"✓ OpenRouter configured with {len(self.api_keys)} API key(s)")
        print(f"✓ Using model: {self.model_name} (with {len(self.models)} fallback models)")
    
    def get_current_api_key(self):
        """Get the current API key"""
        return self.api_keys[self.current_key_index]
    
    def rotate_api_key(self):
        """Rotate to the next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(f"🔄 Rotated to API key #{self.current_key_index + 1}")
    
    def rotate_model(self):
        """Rotate to the next available model"""
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        self.model_name = self.models[self.current_model_index]
        print(f"🔄 Switched to model: {self.model_name}")
    
    def make_request(self, messages, system_message=None, retry_count=0, model_retry_count=0):
        """Make a request to OpenRouter with automatic key and model rotation on failure"""
        
        if retry_count >= len(self.api_keys) * 2:  # Try each key twice
            if model_retry_count >= len(self.models):
                raise Exception("All API keys and models exhausted or unavailable")
            # Try next model
            self.rotate_model()
            return self.make_request(messages, system_message, 0, model_retry_count + 1)
        
        headers = {
            "Authorization": f"Bearer {self.get_current_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/ai-personalization",  # Optional
            "X-Title": "AI Hyper-Personalization Engine"  # Optional
        }
        
        # Prepare messages with system message if provided
        formatted_messages = []
        if system_message:
            formatted_messages.append({
                "role": "system",
                "content": system_message
            })
        formatted_messages.extend(messages)
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:  # Model not available or privacy policy issue
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                print(f"⚠ Model unavailable: {error_msg}")
                # Try next model immediately
                if model_retry_count < len(self.models) - 1:
                    self.rotate_model()
                    time.sleep(0.5)
                    return self.make_request(messages, system_message, 0, model_retry_count + 1)
                else:
                    raise Exception(f"All models unavailable. Error: {error_msg}")
            elif response.status_code == 429:  # Rate limit
                print(f"⚠ Rate limit hit on API key #{self.current_key_index + 1}")
                self.rotate_api_key()
                time.sleep(1)
                return self.make_request(messages, system_message, retry_count + 1, model_retry_count)
            elif response.status_code == 401:  # Invalid key
                print(f"⚠ Invalid API key #{self.current_key_index + 1}")
                self.rotate_api_key()
                return self.make_request(messages, system_message, retry_count + 1, model_retry_count)
            else:
                print(f"⚠ Error {response.status_code}: {response.text}")
                self.rotate_api_key()
                time.sleep(1)
                return self.make_request(messages, system_message, retry_count + 1, model_retry_count)
                
        except Exception as e:
            print(f"⚠ Request failed: {e}")
            if retry_count < len(self.api_keys) - 1:
                self.rotate_api_key()
                time.sleep(1)
                return self.make_request(messages, system_message, retry_count + 1, model_retry_count)
            elif model_retry_count < len(self.models) - 1:
                # Try next model
                self.rotate_model()
                time.sleep(1)
                return self.make_request(messages, system_message, 0, model_retry_count + 1)
            else:
                raise


# ==============================================================================
# AI AGENT WRAPPER FOR OPENROUTER
# ==============================================================================

class AIAgent:
    """Wrapper for OpenRouter to simulate agent behavior"""
    
    def __init__(self, name: str, system_message: str, config: OpenRouterConfig):
        self.name = name
        self.system_message = system_message
        self.config = config
        self.chat_history = []
    
    def analyze(self, prompt: str) -> str:
        """Send a prompt to the agent and get response"""
        try:
            # Add user message to history
            self.chat_history.append({
                "role": "user",
                "content": prompt
            })
            
            # Make request to OpenRouter
            response = self.config.make_request(
                messages=self.chat_history,
                system_message=self.system_message
            )
            
            # Extract response text
            assistant_message = response['choices'][0]['message']['content']
            
            # Add assistant response to history
            self.chat_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except Exception as e:
            print(f"⚠ Error in {self.name}: {e}")
            return f"Analysis unavailable due to error: {str(e)}"
    
    def reset_chat(self):
        """Reset the chat history"""
        self.chat_history = []


# ==============================================================================
# AI-POWERED HYPER-PERSONALIZATION ENGINE
# ==============================================================================

class HyperPersonalizationEngine:
    """
    AI-First Individual-Centric Hyper-Personalization & Churn Intelligence Platform
    
    This system creates individual customer profiles (not personas) and uses AI agents
    to continuously analyze behavior, predict intent, and generate personalized actions.
    """
    
    def __init__(self, data_path, use_ai_agents=True, api_keys=None):
        self.data_path = data_path
        self.df = None
        self.customer_profiles = {}  # Individual customer intelligence
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.use_ai = use_ai_agents
        
        # Initialize AI agents with OpenRouter
        if self.use_ai:
            try:
                self.config = OpenRouterConfig(api_keys=api_keys)
                self.setup_ai_agents()
                print("✓ AI-First Personalization Engine Initialized (OpenRouter)")
            except Exception as e:
                print(f"⚠ AI agents unavailable: {e}")
                self.use_ai = False
    
    def setup_ai_agents(self):
        """Setup specialized AI agents for hyper-personalization"""
        
        # Intent Prediction Agent
        self.intent_agent = AIAgent(
            name="IntentPredictor",
            system_message="""You are an AI that predicts customer intent and next actions.
            
            Analyze customer behavior patterns and predict:
            1. What the customer is likely to do next (upgrade, downgrade, churn, stay)
            2. Why they might take that action (pain points, satisfaction drivers)
            3. When they're most likely to act (urgency signals)
            4. What offer would resonate most with this specific individual
            
            Provide JSON format responses with specific predictions and confidence scores.
            Focus on individual-level insights, not generic personas.""",
            config=self.config
        )
        
        # Propensity Scoring Agent
        self.propensity_agent = AIAgent(
            name="PropensityScorer",
            system_message="""You are an AI that calculates purchase/upsell propensity.
            
            For each customer, analyze:
            1. Likelihood to purchase additional services (0-100%)
            2. Which specific products/services they'd buy
            3. Optimal pricing strategy for this individual
            4. Best time/channel to make the offer
            5. Predicted revenue impact
            
            Provide actionable propensity scores with specific product recommendations.
            Tailor everything to the individual customer's context.""",
            config=self.config
        )
        
        # Churn Prevention Agent
        self.churn_agent = AIAgent(
            name="ChurnPreventer",
            system_message="""You are an AI that predicts and prevents customer churn.
            
            For each at-risk customer:
            1. Identify specific churn risk factors for THIS customer
            2. Predict churn probability and timeline
            3. Generate personalized retention offers
            4. Recommend proactive interventions
            5. Calculate customer lifetime value at risk
            
            Create individual retention strategies, not generic campaigns.
            Focus on preventing churn before it happens.""",
            config=self.config
        )
        
        # Personalization Orchestrator
        self.orchestrator_agent = AIAgent(
            name="PersonalizationOrchestrator",
            system_message="""You are an AI that orchestrates personalized customer experiences.
            
            Synthesize insights from intent, propensity, and churn predictions to:
            1. Create a unified personalization strategy for each customer
            2. Prioritize actions (retain vs upsell vs cross-sell vs nurture)
            3. Generate personalized messaging and offers
            4. Recommend optimal timing and channels
            5. Predict overall business impact
            
            Create real-time, adaptive strategies for individual customers.
            Balance business goals with customer experience.""",
            config=self.config
        )
    
    def load_and_prepare_data(self):
        """Load data and create individual customer profiles"""
        print("\n" + "=" * 80)
        print("LOADING CUSTOMER DATA & CREATING INDIVIDUAL PROFILES")
        print("=" * 80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} customers")
        
        # Store original customer IDs
        if 'customerID' in self.df.columns:
            self.customer_ids = self.df['customerID'].copy()
            self.df_with_ids = self.df.copy()
            self.df = self.df.drop(['customerID'], axis=1)
        
        # Preprocess
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df.dropna(inplace=True)
        self.df["SeniorCitizen"] = self.df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = le.fit_transform(self.df[col])
        
        print(f"✓ Preprocessed data: {self.df.shape}")
        
        # Create individual profiles for each customer
        print("\n📊 Creating individual customer intelligence profiles...")
        self.create_customer_profiles()
        
        return self.df
    
    def create_customer_profiles(self):
        """Create detailed individual profiles for each customer (not personas)"""
        print(f"Creating {len(self.df)} individual customer profiles...")
        
        for idx, row in self.df.iterrows():
            customer_id = self.customer_ids.iloc[idx] if hasattr(self, 'customer_ids') else f"CUST_{idx}"
            
            # Create individual profile
            profile = {
                'customer_id': customer_id,
                'features': row.to_dict(),
                'segment': None,  # Will be assigned by clustering
                'churn_risk': None,
                'churn_probability': None,
                'propensity_scores': {},
                'predicted_intent': None,
                'personalized_strategy': None,
                'lifetime_value': None,
                'risk_factors': [],
                'opportunities': [],
                'next_best_actions': [],
                'created_at': datetime.now().isoformat()
            }
            
            self.customer_profiles[customer_id] = profile
        
        print(f"✓ Created {len(self.customer_profiles)} individual customer profiles")
    
    def build_ml_foundation(self):
        """Build ML models for predictions (foundation for AI agents)"""
        print("\n" + "=" * 80)
        print("BUILDING ML PREDICTION MODELS")
        print("=" * 80)
        
        X = self.df.drop(columns=['Churn'])
        y = self.df['Churn'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = self.scaler.transform(X_test[num_cols])
        
        # Train churn prediction model
        print("\n1. Training Churn Prediction Model...")
        churn_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        churn_model.fit(X_train, y_train)
        accuracy = churn_model.score(X_test, y_test)
        print(f"   ✓ Churn Model Accuracy: {accuracy:.2%}")
        
        # Train propensity models (simulate multiple product propensities)
        print("\n2. Training Propensity Models...")
        propensity_model = RandomForestClassifier(random_state=42, n_estimators=100)
        propensity_model.fit(X_train, y_train)
        print(f"   ✓ Propensity Models Trained")
        
        # Customer segmentation (micro-segments, not broad personas)
        print("\n3. Creating Micro-Segments...")
        kmeans = KMeans(n_clusters=20, random_state=42)  # 20 micro-segments
        X_scaled = self.scaler.fit_transform(X)
        segments = kmeans.fit_predict(X_scaled)
        print(f"   ✓ Created 20 micro-segments (not personas)")
        
        self.ml_models = {
            'churn': churn_model,
            'propensity': propensity_model,
            'segmentation': kmeans,
            'X_scaled': X_scaled
        }
        
        return self.ml_models
    
    def enrich_profiles_with_predictions(self):
        """Enrich individual profiles with ML predictions"""
        print("\n" + "=" * 80)
        print("ENRICHING INDIVIDUAL PROFILES WITH PREDICTIONS")
        print("=" * 80)
        
        X = self.df.drop(columns=['Churn'])
        X_scaled = self.ml_models['X_scaled']
        
        for idx, (customer_id, profile) in enumerate(self.customer_profiles.items()):
            # Get predictions for this individual
            customer_features = X.iloc[idx:idx+1]
            customer_scaled = X_scaled[idx:idx+1]
            
            # Churn prediction
            churn_prob = self.ml_models['churn'].predict_proba(customer_features)[0][1]
            churn_risk = 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
            
            # Propensity scores (simulate for different products)
            base_propensity = self.ml_models['propensity'].predict_proba(customer_features)[0][1]
            
            # Segment assignment
            segment = self.ml_models['segmentation'].predict(customer_scaled)[0]
            
            # Update profile
            profile['churn_probability'] = float(churn_prob)
            profile['churn_risk'] = churn_risk
            profile['segment'] = int(segment)
            profile['propensity_scores'] = {
                'upsell': float(base_propensity * 0.9),
                'cross_sell': float(base_propensity * 0.8),
                'premium_upgrade': float(base_propensity * 0.7),
                'addon_services': float(base_propensity * 0.85)
            }
            
            # Calculate CLV (simplified)
            monthly_charges = profile['features'].get('MonthlyCharges', 50)
            tenure = profile['features'].get('tenure', 12)
            profile['lifetime_value'] = float(monthly_charges * (tenure + 12 * (1 - churn_prob)))
        
        print(f"✓ Enriched {len(self.customer_profiles)} customer profiles with predictions")
    
    def analyze_customer_with_ai(self, customer_id, profile):
        """Use AI agents to deeply analyze an individual customer"""
        
        if not self.use_ai:
            return self.generate_rule_based_strategy(profile)
        
        # Create customer summary for AI analysis
        customer_summary = f"""
INDIVIDUAL CUSTOMER ANALYSIS REQUEST
Customer ID: {customer_id}
=====================================

PROFILE DATA:
- Churn Risk: {profile['churn_risk']} ({profile['churn_probability']:.1%} probability)
- Customer Lifetime Value: ${profile['lifetime_value']:.2f}
- Tenure: {profile['features'].get('tenure', 'N/A')} months
- Monthly Charges: ${profile['features'].get('MonthlyCharges', 'N/A')}
- Contract Type: {profile['features'].get('Contract', 'N/A')}
- Internet Service: {profile['features'].get('InternetService', 'N/A')}
- Segment: Micro-segment #{profile['segment']}

PROPENSITY SCORES:
- Upsell: {profile['propensity_scores']['upsell']:.1%}
- Cross-sell: {profile['propensity_scores']['cross_sell']:.1%}
- Premium Upgrade: {profile['propensity_scores']['premium_upgrade']:.1%}
- Add-on Services: {profile['propensity_scores']['addon_services']:.1%}

TASK:
Analyze this SPECIFIC individual customer and provide:
1. Predicted customer intent and next likely action
2. Personalized retention/upsell strategy
3. Specific product/service recommendations
4. Optimal timing and channel for engagement
5. Expected revenue impact

Respond in JSON format with your analysis.
"""
        
        try:
            # Sequential AI agent analysis
            print(f"      → Intent Agent analyzing...")
            intent_response = self.intent_agent.analyze(customer_summary)
            
            print(f"      → Propensity Agent analyzing...")
            propensity_response = self.propensity_agent.analyze(customer_summary)
            
            print(f"      → Churn Agent analyzing...")
            churn_response = self.churn_agent.analyze(customer_summary)
            
            # Synthesize insights with orchestrator
            print(f"      → Orchestrator synthesizing...")
            synthesis_prompt = f"""
Based on the following AI agent analyses for customer {customer_id}, create a unified personalization strategy:

INTENT ANALYSIS:
{intent_response}

PROPENSITY ANALYSIS:
{propensity_response}

CHURN ANALYSIS:
{churn_response}

Synthesize these into a cohesive strategy with:
1. Primary intent prediction
2. Top 3 recommended actions
3. Retention strategy (if needed)
4. Upsell recommendations (if appropriate)
5. Optimal engagement plan

Return as JSON with keys: intent, next_actions, retention_strategy, upsell_recommendations, engagement_plan
"""
            orchestrator_response = self.orchestrator_agent.analyze(synthesis_prompt)
            
            # Extract structured insights
            ai_insights = self.parse_ai_response(orchestrator_response, profile)
            
            # Reset chat histories for next customer
            self.intent_agent.reset_chat()
            self.propensity_agent.reset_chat()
            self.churn_agent.reset_chat()
            self.orchestrator_agent.reset_chat()
            
            return ai_insights
            
        except Exception as e:
            print(f"⚠ AI analysis failed for {customer_id}: {e}")
            return self.generate_rule_based_strategy(profile)
    
    def parse_ai_response(self, response_text: str, profile: dict) -> dict:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    'intent': parsed.get('intent', 'Analyzed'),
                    'next_actions': parsed.get('next_actions', [])[:3],
                    'retention_strategy': parsed.get('retention_strategy', 'AI-generated'),
                    'upsell_recommendations': parsed.get('upsell_recommendations', []),
                    'engagement_plan': parsed.get('engagement_plan', {}),
                    'ai_powered': True
                }
        except:
            pass
        
        # Fallback: extract key information from text
        insights = {
            'intent': 'Analyzed by AI',
            'next_actions': [],
            'retention_strategy': None,
            'upsell_recommendations': [],
            'engagement_plan': {},
            'ai_powered': True
        }
        
        response_lower = response_text.lower()
        
        # Extract intent
        if 'churn' in response_lower or 'leave' in response_lower:
            insights['intent'] = 'At Risk'
        elif 'upgrade' in response_lower or 'upsell' in response_lower:
            insights['intent'] = 'Growth Opportunity'
        else:
            insights['intent'] = 'Maintain'
        
        # Extract actions based on risk
        if profile['churn_risk'] == 'High':
            insights['next_actions'] = [
                'Immediate retention offer',
                'Personal account manager outreach',
                'Special loyalty program enrollment'
            ]
            insights['retention_strategy'] = 'AI-powered high-priority retention'
        elif profile['churn_risk'] == 'Medium':
            insights['next_actions'] = [
                'Proactive satisfaction check',
                'Service enhancement offer',
                'Engagement campaign'
            ]
        else:
            insights['next_actions'] = [
                'Upsell premium features',
                'Cross-sell complementary services',
                'VIP program invitation'
            ]
            if max(profile['propensity_scores'].values()) > 0.6:
                insights['upsell_recommendations'] = [
                    k for k, v in profile['propensity_scores'].items() if v > 0.6
                ]
        
        return insights
    
    def generate_rule_based_strategy(self, profile):
        """Fallback: Generate personalized strategy using rules"""
        strategy = {
            'intent': 'Stay' if profile['churn_probability'] < 0.3 else 'At Risk',
            'next_actions': [],
            'retention_strategy': None,
            'upsell_recommendations': [],
            'engagement_plan': {},
            'ai_powered': False
        }
        
        # Determine actions based on profile
        if profile['churn_risk'] == 'High':
            strategy['next_actions'] = [
                'Immediate retention offer',
                'Personal outreach from account manager',
                'Special loyalty discount'
            ]
            strategy['retention_strategy'] = f"High-priority retention: Offer ${profile['features'].get('MonthlyCharges', 50) * 0.2:.2f} discount"
        elif profile['churn_risk'] == 'Medium':
            strategy['next_actions'] = [
                'Send satisfaction survey',
                'Offer service upgrade trial',
                'Check for service issues'
            ]
        else:
            # Low churn risk - focus on growth
            if max(profile['propensity_scores'].values()) > 0.6:
                strategy['next_actions'] = [
                    'Upsell premium services',
                    'Cross-sell complementary products'
                ]
                strategy['upsell_recommendations'] = [
                    k for k, v in profile['propensity_scores'].items() if v > 0.6
                ]
        
        return strategy
    
    def run_hyper_personalization(self, num_customers=10):
        """
        Run AI-powered hyper-personalization for individual customers
        
        This demonstrates the AI-first approach: each customer gets individual
        analysis and personalized strategies, not generic persona-based treatment
        """
        print("\n" + "🤖" * 40)
        print("AI-POWERED HYPER-PERSONALIZATION ENGINE")
        print("Individual-Centric Intelligence (Not Persona-Based)")
        print("🤖" * 40)
        
        results = []
        
        # Analyze a sample of customers individually
        customer_sample = list(self.customer_profiles.items())[:num_customers]
        
        print(f"\n🔍 Analyzing {len(customer_sample)} individual customers with AI agents...")
        print("=" * 80)
        
        for idx, (customer_id, profile) in enumerate(customer_sample, 1):
            print(f"\n[{idx}/{len(customer_sample)}] Analyzing Customer: {customer_id}")
            print("-" * 80)
            
            # Display customer snapshot
            print(f"  Churn Risk: {profile['churn_risk']} ({profile['churn_probability']:.1%})")
            print(f"  CLV: ${profile['lifetime_value']:.2f}")
            print(f"  Segment: #{profile['segment']}")
            
            # AI-powered individual analysis
            if self.use_ai and idx <= 3:  # Analyze first 3 with AI (to save API calls)
                print(f"  🤖 Running AI agent analysis...")
                ai_strategy = self.analyze_customer_with_ai(customer_id, profile)
            else:
                print(f"  📊 Generating rule-based strategy...")
                ai_strategy = self.generate_rule_based_strategy(profile)
            
            # Update profile with personalized strategy
            profile['personalized_strategy'] = ai_strategy
            profile['next_best_actions'] = ai_strategy.get('next_actions', [])
            
            # Display strategy
            print(f"  ✓ Strategy Generated:")
            print(f"    Intent: {ai_strategy.get('intent', 'N/A')}")
            actions = ai_strategy.get('next_actions', ['None'])
            print(f"    Actions: {', '.join(actions[:2]) if actions else 'None'}")
            
            results.append({
                'customer_id': customer_id,
                'profile': profile,
                'strategy': ai_strategy
            })
        
        print("\n" + "=" * 80)
        print(f"✅ COMPLETED: {len(results)} customers analyzed individually")
        print("=" * 80)
        
        return results
    
    def generate_business_impact_report(self, results):
        """Generate business impact analysis"""
        print("\n" + "=" * 80)
        print("BUSINESS IMPACT ANALYSIS")
        print("=" * 80)
        
        total_customers = len(results)
        high_risk = sum(1 for r in results if r['profile']['churn_risk'] == 'High')
        medium_risk = sum(1 for r in results if r['profile']['churn_risk'] == 'Medium')
        total_clv_at_risk = sum(r['profile']['lifetime_value'] 
                               for r in results 
                               if r['profile']['churn_risk'] in ['High', 'Medium'])
        
        avg_churn_prob = np.mean([r['profile']['churn_probability'] for r in results])
        
        print(f"\n📊 CUSTOMER RISK DISTRIBUTION:")
        print(f"   High Risk: {high_risk} customers ({high_risk/total_customers*100:.1f}%)")
        print(f"   Medium Risk: {medium_risk} customers ({medium_risk/total_customers*100:.1f}%)")
        print(f"   Low Risk: {total_customers - high_risk - medium_risk} customers")
        
        print(f"\n💰 FINANCIAL IMPACT:")
        print(f"   Total CLV at Risk: ${total_clv_at_risk:,.2f}")
        print(f"   Average Churn Probability: {avg_churn_prob:.1%}")
        print(f"   Potential Saved Revenue (50% retention): ${total_clv_at_risk * 0.5:,.2f}")
        
        print(f"\n🎯 PERSONALIZATION INSIGHTS:")
        print(f"   Individual Profiles Created: {len(self.customer_profiles)}")
        ai_powered = sum(1 for r in results if r['strategy'].get('ai_powered', False))
        print(f"   AI-Powered Strategies: {ai_powered}")
        print(f"   Micro-Segments Identified: 20 (vs typical 5-10 personas)")
        
        return {
            'total_customers': total_customers,
            'high_risk': high_risk,
            'clv_at_risk': total_clv_at_risk,
            'potential_savings': total_clv_at_risk * 0.5
        }
    
    def run_full_pipeline(self, analyze_customers=10):
        """Run the complete hyper-personalization pipeline"""
        print("\n" + "🚀" * 40)
        print("AI-DRIVEN HYPER-PERSONALIZATION & CHURN INTELLIGENCE PLATFORM")
        print("🚀" * 40)
        
        # 1. Load data and create individual profiles
        self.load_and_prepare_data()
        
        # 2. Build ML foundation
        self.build_ml_foundation()
        
        # 3. Enrich profiles with predictions
        self.enrich_profiles_with_predictions()
        
        # 4. Run AI-powered hyper-personalization
        results = self.run_hyper_personalization(num_customers=analyze_customers)
        
        # 5. Generate business impact report
        impact = self.generate_business_impact_report(results)
        
        print("\n" + "✅" * 40)
        print("PIPELINE COMPLETED!")
        print("✅" * 40)
        
        print("\n💡 KEY DIFFERENTIATORS FROM TRADITIONAL APPROACHES:")
        print("   ✓ Individual customer profiles (not generic personas)")
        print("   ✓ AI agents analyze each customer separately")
        print("   ✓ Real-time personalized strategies")
        print("   ✓ Intent prediction at individual level")
        print("   ✓ Dynamic propensity scoring")
        print("   ✓ Proactive churn prevention with personalized offers")
        
        return {
            'results': results,
            'impact': impact,
            'total_profiles': len(self.customer_profiles)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AI-DRIVEN HYPER-PERSONALIZATION & CHURN INTELLIGENCE PLATFORM")
    print("=" * 80)
    print("\nSolving: Individual-level behavior prediction vs persona-based segmentation")
    print("Approach: AI-first, adaptive, real-time personalization at scale")
    print("Powered by: OpenRouter (Free Models) with API key rotation")
    print("\n" + "=" * 80)
    
    # OPTION 1: Use environment variables (recommended)
    # Set these in your .env file or environment:
    # OPENROUTER_API_KEY_1=sk-or-v1-...
    # OPENROUTER_API_KEY_2=sk-or-v1-...
    # OPENROUTER_API_KEY_3=sk-or-v1-...
    
    engine = HyperPersonalizationEngine(
        data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv',
        use_ai_agents=True,  # Set to False for faster testing without AI
        api_keys=[
              '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',
              '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',
              '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'
         ]  # Will use environment variables
    )
    
    # OPTION 2: Pass API keys directly (less secure, not recommended for production)
    # engine = HyperPersonalizationEngine(
    #     data_path='WA_Fn-UseC_-Telco-Customer-Churn.csv',
    #     use_ai_agents=True,
    #     api_keys=[
    #         'sk-or-v1-your-key-1-here',
    #         'sk-or-v1-your-key-2-here',
    #         'sk-or-v1-your-key-3-here'
    #     ]
    # )
    
    # Run the full pipeline
    # analyze_customers: number of customers to deeply analyze with AI (set lower for faster demo)
    results = engine.run_full_pipeline(analyze_customers=5)