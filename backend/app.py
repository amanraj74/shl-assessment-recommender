from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import AssessmentRecommender
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize recommender
try:
    recommender = AssessmentRecommender()
    print("✅ Recommender initialized successfully")
except Exception as e:
    print(f"❌ Error initializing recommender: {e}")
    recommender = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "SHL Assessment Recommendation API is running",
        "version": "2.0"
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """Assessment recommendation endpoint"""
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400
        
        query = data['query']
        k = data.get('top_k', 10)
        
        # Validate k
        if k < 5 or k > 10:
            k = 10
        
        # Get recommendations
        recommendations = recommender.get_recommendations(query, k=k)
        
        response = {
            "query": query,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error in /recommend: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/batch_recommend', methods=['POST'])
def batch_recommend():
    """Batch recommendation for test set"""
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 500
        
        data = request.get_json()
        queries = data.get('queries', [])
        
        results = []
        for query in queries:
            recommendations = recommender.get_recommendations(query, k=10)
            results.append({
                'query': query,
                'recommendations': recommendations
            })
        
        return jsonify({"results": results}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
