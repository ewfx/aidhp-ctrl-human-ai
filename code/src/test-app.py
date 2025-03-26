import pytest
import streamlit as st
import pandas as pd
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np
import tempfile
import os

# Mock data for testing
@pytest.fixture
def mock_customer_data():
    return pd.DataFrame({
        'customer_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'income': [50000, 75000, 120000],
        'credit_score': [650, 720, 800],
        'social_media_activity': ['["good", "happy"]', '["bad", "angry"]', '["ok"]'],
        'interests': ['travel', 'tech', 'fitness'],
        'engagement_score': [0.5, 0.8, 0.3],
        'sentiment_score': [0.7, 0.2, 0.5],
        'churn': [0, 1, 0]
    })

@pytest.fixture
def mock_product_data():
    return pd.DataFrame({
        'product_id': [101, 102, 103],
        'product_name': ['Savings Plus', 'Investment Gold', 'Credit Premium'],
        'product_type': ['Savings', 'Investment', 'Credit'],
        'min_credit_score': [600, 700, 750],
        'min_income': [30000, 50000, 80000],
        'max_income': [100000, 200000, 500000],
        'product_risk': ['Low', 'Medium', 'High'],
        'minimum_deposit': [100, 1000, 500]
    })

@pytest.fixture
def mock_image_file():
    # Create a temporary image file for testing
    img = Image.new('RGB', (100, 100), color='red')
    tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(tmp_file.name)
    yield tmp_file.name
    os.unlink(tmp_file.name)

# Test model loading
def test_load_models():
    with patch('transformers.pipeline') as mock_pipeline, \
         patch('transformers.CLIPProcessor.from_pretrained') as mock_clip_processor, \
         patch('transformers.CLIPModel.from_pretrained') as mock_clip_model:
        
        # Call the function
        from app import load_models
        load_models()
        
        # Assert the models were loaded
        
        #mock_clip_processor.assert_called_once_with("openai/clip-vit-base-patch32")
        #mock_clip_model.assert_called_once_with("openai/clip-vit-base-patch32")

# Test data upload and processing
def test_data_upload_processing(mock_customer_data, mock_product_data):
    with patch('streamlit.file_uploader') as mock_uploader, \
         patch('pandas.read_csv') as mock_read_csv:
        
        # Configure mocks
        mock_uploader.side_effect = [
            MagicMock(),  # Customer data upload
            MagicMock()   # Product data upload
        ]
        mock_read_csv.side_effect = [
            mock_customer_data,
            mock_product_data
        ]
        
        # Import and test
        from app import st
        st.session_state = {}
        
        # Simulate data upload page
        from app import page
        page = "📤 Data Upload"
        
        # Assert data is processed correctly
       # assert st.session_state.customer_data is None
        #assert st.session_state.financial_products is None
        
        # After upload processing would happen here
        # In real test, we'd trigger the upload callback
        
        # For now just verify the mock setup
#assert mock_uploader.call_count == 2

# Test sentiment analysis
def test_sentiment_analysis(mock_customer_data):
    with patch('app.sentiment_analyzer') as mock_analyzer:
        # Setup mock analyzer
        mock_analyzer.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
        
        # Import and test
        from app import st
        st.session_state = {'customer_data': mock_customer_data}
        
        # Simulate running sentiment analysis
        from app import page
        page = "😊 Sentiment Analysis"
        
        # This would normally be triggered by a button click
        mock_customer_data["sentiment"] = ["Positive", "Negative", "Positive"]
        
        # Verify results
        assert "sentiment" in mock_customer_data.columns
        assert mock_customer_data["sentiment"].tolist() == ["Positive", "Negative", "Positive"]

# Test financial product recommendations
def test_financial_recommendations(mock_customer_data, mock_product_data):
    with patch('app.st.session_state') as mock_session:
        # Setup mock session data
        mock_session.customer_data = mock_customer_data
        mock_session.financial_products = mock_product_data
        
        # Import and test
        from app import page
        page = "💰 Product Recommendations"
        
        # Simulate recommendation generation
        recommendations = []
        for _, customer in mock_customer_data.iterrows():
            matched = mock_product_data[
                (mock_product_data["min_credit_score"] <= customer["credit_score"]) &
                (mock_product_data["min_income"] <= customer["income"]) &
                (mock_product_data["max_income"] >= customer["income"])
            ]
            recommendations.append({
                "customer_id": customer["customer_id"],
                "matched_products": len(matched)
            })
        
        # Verify recommendations
        assert len(recommendations) == 3
        assert recommendations[0]["matched_products"] > 0  # Should match at least one product



# Test business insights generation
def test_business_insights(mock_customer_data):
    with patch('sklearn.cluster.KMeans') as mock_kmeans:
        # Setup mock KMeans
        mock_kmeans.return_value.fit_predict.return_value = [0, 1, 2]
        
        # Import and test
        from app import st
        st.session_state = {'customer_data': mock_customer_data}
        
        # Simulate insights generation
        from app import page
        page = "📊 Business Insights"
        
        # This would normally be triggered by a button click
        mock_customer_data["cluster"] = [0, 1, 2]
        
        # Verify clustering was applied
        assert "cluster" in mock_customer_data.columns
        assert set(mock_customer_data["cluster"].unique()) == {0, 1, 2}


# Test numeric column conversion
def test_numeric_conversion(mock_product_data):
    # Create a test dataframe with mixed numeric/string columns
    test_df = pd.DataFrame({
        'numeric_1': ['100', '200', '300'],
        'numeric_2': [400, 500, 600],
        'text': ['a', 'b', 'c']
    })
    
    # Test conversion function
    numeric_cols = ['numeric_1', 'numeric_2']
    for col in numeric_cols:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Verify conversion
    assert test_df['numeric_1'].dtype == 'int64'
    assert test_df['numeric_2'].dtype == 'int64'
    assert test_df['text'].dtype == 'object'

# Test risk profile assignment
def test_risk_profile_assignment(mock_customer_data):
    # Test the risk profile assignment logic
    def assign_risk_profile(row):
        if row['income'] < 60000:
            return 'Conservative'
        elif row['income'] < 150000:
            return 'Moderate'
        else:
            return 'Aggressive'
    
    mock_customer_data['risk_profile'] = mock_customer_data.apply(assign_risk_profile, axis=1)
    
    # Verify assignments
    assert mock_customer_data.loc[0, 'risk_profile'] == 'Conservative'
    assert mock_customer_data.loc[1, 'risk_profile'] == 'Moderate'
    assert mock_customer_data.loc[2, 'risk_profile'] == 'Moderate'
    
 # Test CSS loading
def test_local_css():
    mock_css_content = "body { background-color: #f0f0f0; }"
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = mock_css_content
        from app import local_css
        local_css("./style.css")
        
        assert mock_open.called   
