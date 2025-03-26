import streamlit as st
import pandas as pd
import tempfile
from transformers.pipelines import pipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from PIL import Image

# Set page config
st.set_page_config(
    page_title="💰 AI-Driven Hyper-Personalization & Recommendations Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Load models
@st.cache_resource
def load_models():
    try:
        # Try standard import first
        from transformers import pipeline, CLIPProcessor, CLIPModel
    except ImportError:
        # Fallback imports if standard import fails
        from transformers.pipelines import pipeline
        from transformers.models.clip import CLIPProcessor, CLIPModel
    
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return sentiment_analyzer, clip_processor, clip_model

sentiment_analyzer, clip_processor, clip_model = load_models()

# Initialize session state
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'financial_products' not in st.session_state:
    st.session_state.financial_products = None
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

st.markdown("""
<div class="header">
    <h1>🌟 AI-Driven Hyper-Personalization & Recommendations Dashboard</h1>
    <p>Hyper-personalized recommendations and insights powered by <strong style="color:#9d50bb">Ctrl-Human-AI Advisor</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["📤 Data Upload", "😊 Sentiment Analysis", "💡 Recommendations", 
         "🔮 Predictive Insights", "🖼️ Multi-Modal Recommendations", "📊 Business Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div class="sidebar-footer">
        <p>Powered by:</p>
        <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="30">
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="30">
    </div>
    """, unsafe_allow_html=True)

# Data Upload Page
if page == "📤 Data Upload":
    st.markdown("""
    <div class="section-header">
        <h2>📤 Data Upload Center</h2>
        <p>Upload your customer and product data to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class="card">
                <h3>👥 Customer Data</h3>
            </div>
            """, unsafe_allow_html=True)
            customer_file = st.file_uploader("Upload Customer Data (CSV)", type=["csv"], key="customer_upload")
            if customer_file is not None:
                customer_data = pd.read_csv(customer_file)
                st.session_state.customer_data = customer_data
                st.success("✅ Customer data uploaded successfully!")
                with st.expander("👀 Preview Customer Data"):
                    st.dataframe(customer_data.head().style.highlight_max(axis=0))
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="card">
                <h3>💳 Financial Products</h3>
            </div>
            """, unsafe_allow_html=True)
            products_file = st.file_uploader("Upload Financial Products (CSV)", type=["csv"], key="products_upload")
            if products_file is not None:
                try:
                    # Read and convert numeric columns
                    financial_products = pd.read_csv(products_file)
                
                    # List of expected numeric columns
                    numeric_cols = ['min_credit_score', 'min_income', 'max_income', 
                                  'annual_fee', 'interest_rate', 'minimum_deposit']
                
                    # Convert to numeric, coerce errors to NaN
                    for col in numeric_cols:
                        if col in financial_products.columns:
                            financial_products[col] = pd.to_numeric(financial_products[col], errors='coerce')
                
                    st.session_state.financial_products = financial_products
                    st.success("✅ Financial products data uploaded successfully!")
                
                    with st.expander("👀 Preview Product Data"):
                        # Display without style if conversion fails
                        try:
                            st.dataframe(financial_products.head().style.highlight_max(axis=0))
                        except:
                            st.dataframe(financial_products.head())
                            st.warning("Couldn't apply styling - some numeric columns contain text")
                        
                except Exception as e:
                    st.error(f"❌ Error loading financial products: {str(e)}")
    
        with st.container():
            st.markdown("""
            <div class="card">
                <h3>🖼️ Product Images</h3>
            </div>
            """, unsafe_allow_html=True)
            image_files = st.file_uploader("Upload Product Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if image_files:
                uploaded_paths = []
                for image_file in image_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=image_file.name) as tmp_file:
                        tmp_file.write(image_file.read())
                        uploaded_paths.append(tmp_file.name)
                st.session_state.uploaded_images = uploaded_paths
                st.success(f"✅ {len(uploaded_paths)} images uploaded successfully!")
            
                st.subheader("📸 Uploaded Image Gallery")
                cols = st.columns(3)
                for i, img_path in enumerate(uploaded_paths[:3]):
                    with cols[i]:
                        st.image(img_path, caption=f"Image {i+1}", use_container_width=True)

# Sentiment Analysis Page
elif page == "😊 Sentiment Analysis":
    st.markdown("""
    <div class="section-header">
        <h2>😊 Customer Sentiment Analysis</h2>
        <p>Analyze customer sentiments from social media activity</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.customer_data is None:
        st.warning("⚠️ Please upload customer data first.")
    else:
        if st.button("🚀 Run Sentiment Analysis", help="Click to analyze customer sentiments"):
            with st.spinner("🔍 Analyzing sentiment... This may take a moment"):
                try:
                    if "social_media_activity" not in st.session_state.customer_data.columns:
                        st.error("❌ Column 'social_media_activity' not found in customer data.")
                    else:
                        st.session_state.customer_data["sentiment"] = st.session_state.customer_data["social_media_activity"].apply(
                            lambda posts: "Positive" if sentiment_analyzer(" ".join(eval(posts)))[0]['label'] == "POSITIVE" else "Negative"
                        )
                        
                        st.success("🎉 Sentiment analysis completed!")
                        
                        pos_count = (st.session_state.customer_data["sentiment"] == "Positive").sum()
                        neg_count = (st.session_state.customer_data["sentiment"] == "Negative").sum()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("👍 Positive Sentiments", pos_count)
                        col2.metric("👎 Negative Sentiments", neg_count)
                        
                        with st.expander("📊 View Detailed Results", expanded=True):
                            st.dataframe(
                                st.session_state.customer_data[["customer_id", "social_media_activity", "sentiment"]].head(10)
                                .style.applymap(lambda x: "background-color: #e6f7e6" if x == "Positive" else "background-color: #ffe6e6", subset=["sentiment"])
                            )
                        
                        csv = st.session_state.customer_data.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            data=csv,
                            file_name="customer_data_with_sentiment.csv",
                            mime="text/csv",
                            help="Download the complete analysis results"
                        )
                except Exception as e:
                    st.error(f"❌ Error during sentiment analysis: {str(e)}")

# Recommendations Page
elif page == "💡 Recommendations":
    st.markdown("""
    <div class="section-header">
        <h2>💡 Personalized Recommendations</h2>
        <p>AI-powered financial product matching based on your profile</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.customer_data is None or st.session_state.financial_products is None:
        st.warning("⚠️ Please upload both customer and product data first.")
    else:
        # Clean data
        st.session_state.customer_data.columns = st.session_state.customer_data.columns.str.strip()
        st.session_state.financial_products.columns = st.session_state.financial_products.columns.str.strip()
        
        # Add risk profile if missing
        if 'risk_profile' not in st.session_state.customer_data.columns:
            st.session_state.customer_data['risk_profile'] = st.session_state.customer_data.apply(
                lambda row: 'Conservative' if row['income'] < 30000 
                else 'Moderate' if row['income'] < 75000 
                else 'Aggressive', 
                axis=1
            )
        
        if st.button("✨ Generate Recommendations"):
            with st.spinner("🔍 Analyzing profiles..."):
                try:
                    # ============================================
                    # Core Recommendation Logic
                    # ============================================
                    def generate_recommendations(row, products_df):
                        results = {
                            'perfect_matches': [],
                            'near_matches': [],
                            'alternatives': [],
                            'tips': []
                        }
                        
                        # --------------------------------------------------
                        # 1. Perfect Matches (meets all criteria)
                        # --------------------------------------------------
                        perfect_matches = products_df[
                            (products_df['min_credit_score'] <= row['credit_score']) &
                            (products_df['min_income'] <= row['income']) &
                            (
                                (products_df['max_income'] >= row['income']) | 
                                (products_df['max_income'].isna())
                            )
                        ]
                        
                        # Risk profile filtering
                        if row['risk_profile'] == 'Conservative':
                            perfect_matches = perfect_matches[
                                perfect_matches['product_risk'].isin(['Low', 'Low-Medium'])
                            ]
                        elif row['risk_profile'] == 'Moderate':
                            perfect_matches = perfect_matches[
                                ~perfect_matches['product_risk'].isin(['High', 'Very High'])
                            ]
                        
                        # Interest-based boosting
                        interests = eval(row['interests']) if isinstance(row['interests'], str) else row['interests']
                        interest_boost = 1.0
                        if 'investing' in interests:
                            perfect_matches['score'] = perfect_matches.apply(
                                lambda x: x.get('score', 1) * 1.2 if 'Investment' in x['product_type'] else x.get('score', 1),
                                axis=1
                            )
                        
                        # --------------------------------------------------
                        # 2. Near Matches (slightly outside criteria)
                        # --------------------------------------------------
                        near_matches = products_df[
                            (
                                (products_df['min_credit_score'] <= row['credit_score'] + 30) |
                                (products_df['min_income'] <= row['income'] * 1.3)
                            ) & 
                            ~products_df.index.isin(perfect_matches.index)
                        ]
                        
                        # --------------------------------------------------
                        # 3. Alternatives (same category but easier to qualify)
                        # --------------------------------------------------
                        alternatives = products_df[
                            (products_df['min_credit_score'] <= max(550, row['credit_score'] - 50)) &
                            (products_df['min_income'] <= max(20000, row['income'] * 0.7))
                        ]
                        
                        # --------------------------------------------------
                        # 4. Financial Tips
                        # --------------------------------------------------
                        tips = []
                        if row['credit_score'] < 650:
                            tips.append("💳 Consider secured credit cards to build credit history")
                        if row['income'] < 30000:
                            tips.append("🏦 Look for 'second chance' banking products")
                        if 'retirement' in str(interests).lower() and row['age'] > 40:
                            tips.append("🧓 Schedule a free retirement planning consultation")
                            
                        # --------------------------------------------------
                        # Formatting Logic
                        # --------------------------------------------------
                        def format_product(product):
                            details = [
                                f"<b>{product['product_name']}</b>",
                                f"Type: {product['product_type']}",
                                f"Min Credit: {product['min_credit_score']}+",
                                f"Min Income: ${product['min_income']:,}"
                            ]
                            if product['annual_fee'] > 0:
                                details.append(f"Fee: ${product['annual_fee']}")
                            if product['interest_rate'] > 0:
                                details.append(f"Rate: {product['interest_rate']}%")
                            return " | ".join(details)
                        
                        # Sort and select top 3 for each category
                        perfect_matches = perfect_matches.sort_values(
                            by=['min_credit_score', 'min_income'], 
                            ascending=[False, False]
                        ).head(3)
                        
                        near_matches = near_matches.sort_values(
                            by=['min_credit_score', 'min_income'], 
                            ascending=[False, False]
                        ).head(2)
                        
                        alternatives = alternatives.sort_values(
                            by=['min_credit_score', 'min_income'], 
                            ascending=[True, True]
                        ).head(2)
                        
                        # Build final output
                        output = []
                        
                        if len(perfect_matches) > 0:
                            output.append("<div class='perfect-match'><h4>🎯 Perfect Matches</h4>")
                            output.extend([format_product(p) for _, p in perfect_matches.iterrows()])
                            output.append("</div>")
                        
                        if len(near_matches) > 0:
                            output.append("<div class='near-match'><h4>🌟 Almost There</h4>")
                            for _, p in near_matches.iterrows():
                                if p['min_credit_score'] > row['credit_score']:
                                    output.append(
                                        f"{format_product(p)}<br>"
                                        f"<i>Improve credit score by {p['min_credit_score'] - row['credit_score']} points</i>"
                                    )
                                else:
                                    output.append(
                                        f"{format_product(p)}<br>"
                                        f"<i>Increase income by ${p['min_income'] - row['income']:,}</i>"
                                    )
                            output.append("</div>")
                        
                        if len(alternatives) > 0:
                            output.append("<div class='alternative'><h4>💡 Starter Options</h4>")
                            output.extend([format_product(p) for _, p in alternatives.iterrows()])
                            output.append("</div>")
                        
                        if tips:
                            output.append("<div class='tips'><h4>🔍 Smart Tips</h4>")
                            output.extend([f"• {tip}" for tip in tips])
                            output.append("</div>")
                        
                        if not output:
                            output.append("Contact our advisors for personalized options 📞")
                            
                        return "<br>".join(output)
                    
                    # ============================================
                    # Apply to all customers
                    # ============================================
                    st.session_state.customer_data["recommendations"] = st.session_state.customer_data.apply(
                        lambda row: generate_recommendations(row, st.session_state.financial_products),
                        axis=1
                    )
                    
                    # ============================================
                    # Display Results
                    # ============================================
                    st.success(f"✅ Generated recommendations for {len(st.session_state.customer_data)} customers!")
                    
                    with st.expander("💎 Top Recommendations", expanded=True):
                        for _, row in st.session_state.customer_data.head(3).iterrows():
                            st.markdown(f"""
                            <div class="customer-card">
                                <h3>👤 {row.get('name', f"Customer {row['customer_id']}")}</h3>
                                <p>📊 Age: {row['age']} | Income: ${row['income']:,} | Credit: {row['credit_score']} | Risk: {row['risk_profile']}</p>
                                <p>🎯 Interests: {', '.join(eval(row['interests']) if isinstance(row['interests'], str) else row['interests'])}</p>
                                <div class="recommendations">{row['recommendations']}</div>
                            </div>
                            <hr>
                            """, unsafe_allow_html=True)
                    
                    # Download button
                    csv = st.session_state.customer_data.to_csv(index=False)
                    st.download_button(
                        "📥 Download All Recommendations",
                        data=csv,
                        file_name="personalized_recommendations.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"""❌ Error: {str(e)} 
                    Please check if your data contains all required columns:
                    customer_id, age, income, credit_score, interests, purchase_history""")
# Predictive Insights Page
elif page == "🔮 Predictive Insights":
    st.markdown("""
    <div class="section-header">
        <h2>🔮 Predictive Insights</h2>
        <p>Predict customer churn and other key metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.customer_data is None:
        st.warning("⚠️ Please upload customer data first.")
    else:
        if st.button("🔮 Generate Insights", help="Click to generate predictive insights"):
            with st.spinner("🔮 Analyzing data patterns..."):
                try:
                    required_columns = ['engagement_score', 'sentiment_score', 'churn']
                    missing_cols = [col for col in required_columns if col not in st.session_state.customer_data.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing columns in customer data: {', '.join(missing_cols)}")
                    else:
                        X = st.session_state.customer_data[['engagement_score', 'sentiment_score']]
                        y = st.session_state.customer_data['churn']
                        model = RandomForestClassifier()
                        model.fit(X, y)
                        
                        st.session_state.customer_data["churn_probability"] = model.predict_proba(X)[:, 1]
                        
                        st.success("🎉 Predictive insights generated!")
                        
                        avg_churn_prob = st.session_state.customer_data["churn_probability"].mean()
                        high_risk = (st.session_state.customer_data["churn_probability"] > 0.7).sum()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("📊 Average Churn Probability", f"{avg_churn_prob:.2%}")
                        col2.metric("⚠️ High-Risk Customers", high_risk)
                        
                        with st.expander("📈 Detailed Insights", expanded=True):
                            st.dataframe(
                                st.session_state.customer_data[["customer_id", "churn", "churn_probability"]].head(10)
                                .style.background_gradient(subset=["churn_probability"], cmap="YlOrRd")
                            )
                        
                        csv = st.session_state.customer_data.to_csv(index=False)
                        st.download_button(
                            "📥 Download Insights",
                            data=csv,
                            file_name="customer_data_with_churn_probability.csv",
                            mime="text/csv",
                            help="Download complete insights"
                        )
                except Exception as e:
                    st.error(f"❌ Error generating predictive insights: {str(e)}")

# Multi-Modal Recommendations Page
elif page == "🖼️ Multi-Modal Recommendations":
    st.markdown("""
    <div class="section-header">
        <h2>🖼️ Multi-Modal Recommendations</h2>
        <p>Find visual matches for customer preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.uploaded_images:
        st.warning("⚠️ Please upload product images first.")
    else:
        st.subheader("📸 Uploaded Image Gallery")
        cols = st.columns(3)
        for i, img_path in enumerate(st.session_state.uploaded_images[:3]):
            with cols[i]:
                st.image(img_path, caption=f"Image {i+1}", use_container_width=True)
        
        st.subheader("🔍 Enter Customer Preferences")
        customer_text = st.text_area("Describe what the customer is looking for:")
        
        if st.button("🖼️ Find Best Match", help="Click to find matching products"):
            if not customer_text:
                st.warning("⚠️ Please enter customer preferences first.")
            else:
                with st.spinner("🔍 Finding best visual match..."):
                    try:
                        product_images = []
                        for image_path in st.session_state.uploaded_images:
                            img = Image.open(image_path)
                            product_images.append(img)
                        
                        inputs = clip_processor(text=customer_text, images=product_images, return_tensors="pt", padding=True)
                        outputs = clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        best_match_index = logits_per_image.argmax().item()
                        
                        st.success("🎉 Best matching product found!")
                        
                        st.markdown("""
                        <div class="match-container">
                            <h3>✨ Best Match For:</h3>
                            <p class="customer-query">"{}"</p>
                        </div>
                        """.format(customer_text), unsafe_allow_html=True)
                        
                        st.image(
                            st.session_state.uploaded_images[best_match_index],
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Error processing images: {str(e)}")

# Business Insights Page
elif page == "📊 Business Insights":
    st.markdown("""
    <div class="section-header">
        <h2>📊 Business Insights</h2>
        <p>Customer segmentation and strategic recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.customer_data is None:
        st.warning("⚠️ Please upload customer data first.")
    else:
        if st.button("📈 Generate Insights", help="Click to generate business insights"):
            with st.spinner("🧩 Analyzing customer segments..."):
                try:
                    required_columns = ['engagement_score', 'sentiment_score']
                    missing_cols = [col for col in required_columns if col not in st.session_state.customer_data.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing columns in customer data: {', '.join(missing_cols)}")
                    else:
                        kmeans = KMeans(n_clusters=3)
                        st.session_state.customer_data["cluster"] = kmeans.fit_predict(
                            st.session_state.customer_data[['engagement_score', 'sentiment_score']]
                        )
                        
                        cluster_strategies = {
                            0: "🎯 Target with loyalty programs and exclusive offers",
                            1: "💡 Engage with personalized content and discounts",
                            2: "🔄 Re-engage with win-back campaigns"
                        }
                        insights = st.session_state.customer_data["cluster"].map(cluster_strategies)
                        
                        results_df = pd.DataFrame({
                            "customer_id": st.session_state.customer_data["customer_id"],
                            "cluster": st.session_state.customer_data["cluster"],
                            "strategy": insights
                        })
                        
                        st.success("🎉 Business insights generated!")
                        
                        st.subheader("📊 Customer Segmentation")
                        cluster_counts = results_df["cluster"].value_counts().sort_index()
                        st.bar_chart(cluster_counts)
                        
                        st.subheader("🎯 Recommended Strategies")
                        st.dataframe(
                            results_df.head(10).style.applymap(
                                lambda x: "background-color: #e6f7e6" if x == cluster_strategies[0] 
                                else "background-color: #fff2e6" if x == cluster_strategies[1] 
                                else "background-color: #ffe6e6", subset=["strategy"]))
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Insights",
                            data=csv,
                            file_name="business_insights.csv",
                            mime="text/csv",
                            help="Download complete insights"
                        )
                except Exception as e:
                    st.error(f"❌ Error generating business insights: {str(e)}")

# Financial Product Recommendations
# Financial Product Recommendations (Enhanced UI)
if page == "💰 Product Recommendations":
    st.markdown("""
    <div class="section-header">
        <h2>💰 Financial Product Recommendations</h2>
        <p>AI-powered personalized financial product matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.customer_data is None or st.session_state.financial_products is None:
        st.warning("⚠️ Please upload both customer and financial product data first.")
    else:
        # Add risk profile if missing (demo purposes)
        if 'risk_profile' not in st.session_state.customer_data.columns:
            st.session_state.customer_data['risk_profile'] = st.session_state.customer_data.apply(
                lambda row: 'Conservative' if row['income'] < 60000 
                else 'Moderate' if row['income'] < 150000 
                else 'Aggressive', 
                axis=1
            )
        
        # Display customer profile summary
        with st.expander("👥 Customer Profile Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(st.session_state.customer_data))
            with col2:
                avg_income = st.session_state.customer_data['income'].mean()
                st.metric("Average Income", f"${avg_income:,.0f}")
            with col3:
                common_profile = st.session_state.customer_data['risk_profile'].mode()[0]
                st.metric("Most Common Profile", common_profile)
        
        if st.button("✨ Generate Recommendations", type="primary"):
            with st.spinner("🔍 Analyzing profiles and matching products..."):
                try:
                    # Normalize column names
                    financial_products = st.session_state.financial_products.copy()
                    financial_products.columns = financial_products.columns.str.strip().str.lower().str.replace(" ", "_")
                    
                    # Generate recommendations
                    recommendations = []
                    for _, customer in st.session_state.customer_data.iterrows():
                        # Base matching
                        matched_products = financial_products[
                            (financial_products["min_credit_score"] <= customer["credit_score"]) &
                            (financial_products["min_income"] <= customer["income"]) &
                            (financial_products["max_income"] >= customer["income"])
                        ].copy()
                        
                        # Risk-based filtering
                        if customer['risk_profile'] == 'Conservative':
                            matched_products = matched_products[matched_products['product_risk'] == 'Low']
                        elif customer['risk_profile'] == 'Moderate':
                            matched_products = matched_products[matched_products['product_risk'].isin(['Low', 'Medium'])]
                        
                        # Add special flags
                        matched_products['recommendation_strength'] = matched_products.apply(
                            lambda p: "⭐⭐⭐⭐⭐" if (p['product_risk'] == customer['risk_profile']) else "⭐⭐⭐",
                            axis=1
                        )
                        
                        recommendations.append({
                            "customer_id": customer["customer_id"],
                            "customer_name": customer.get("name", "N/A"),
                            "income": customer["income"],
                            "credit_score": customer["credit_score"],
                            "risk_profile": customer["risk_profile"],
                            "recommended_products": matched_products.to_dict("records")
                        })
                    
                    # Convert to dataframe
                    rec_df = pd.DataFrame(recommendations)
                    
                    # Display results
                    st.success(f"🎉 Generated {len(rec_df)} personalized recommendations!")
                    
                    # Show sample recommendations
                    with st.expander("💎 Top Recommendations Preview", expanded=True):
                        sample_cust = rec_df.iloc[0]
                        st.markdown(f"""
                        <div class="customer-card">
                            <h3>👤 {sample_cust['customer_name']}</h3>
                            <p>💵 Income: ${sample_cust['income']:,} | 📊 Credit: {sample_cust['credit_score']} | 🎯 Profile: {sample_cust['risk_profile']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        cols = st.columns(3)
                        for i, product in enumerate(sample_cust['recommended_products'][:3]):
                            with cols[i % 3]:
                                st.markdown(f"""
                                <div class="product-card">
                                    <h4>{product['product_name']} {product['recommendation_strength']}</h4>
                                    <p>🏷️ Type: {product['product_type']}</p>
                                    <p>📉 Risk: {product['product_risk']}</p>
                                    <p>💵 Min. Deposit: ${product.get('minimum_deposit', 0):,}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Full recommendations table
                    with st.expander("📊 View All Recommendations"):
                        exploded_df = rec_df.explode("recommended_products")
                        normalized_df = pd.json_normalize(exploded_df["recommended_products"])
                        final_df = pd.concat([
                            exploded_df[["customer_id", "customer_name", "risk_profile"]].reset_index(drop=True), 
                            normalized_df
                        ], axis=1)
                        
                        st.dataframe(
                            final_df.head(20).style.apply(
                                lambda x: ['background: #f0f9ff' if x.name%2==0 else '' for i in x], 
                                axis=1
                            )
                        )
                    
                    # Download button
                    csv = final_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Recommendations",
                        data=csv,
                        file_name="financial_product_recommendations.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")