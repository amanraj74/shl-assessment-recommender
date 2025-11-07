"""
V2.0 Flask API Server - Complete Professional Production Version
REST API for assessment recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import AssessmentRecommender
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global recommender instance
recommender = None

def initialize_recommender():
    """Initialize recommender on startup"""
    global recommender
    try:
        logger.info("🏆 Initializing V2.0 Recommender...")
        recommender = AssessmentRecommender()
        logger.info("✅ Recommender initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize recommender: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "message": "V2.0 Assessment Recommendation API is running",
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "recommender_initialized": recommender is not None
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Get assessment recommendations
    
    Request JSON:
    {
        "query": "Job description or search query",
        "top_k": 10  (optional, default: 10)
    }
    
    Response JSON:
    {
        "query": "...",
        "recommendations": [
            {
                "name": "Assessment Name",
                "url": "...",
                "test_type": "...",
                "duration": 0,
                "relevance_score": 0.85
            }
        ],
        "count": 10
    }
    """
    try:
        # Check recommender is initialized
        if recommender is None:
            logger.error("❌ Recommender not initialized")
            return jsonify({
                "error": "Recommender not initialized"
            }), 503
        
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Request body must be JSON"
            }), 400
        
        if 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request body"
            }), 400
        
        # Extract parameters
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        
        # Validate query
        if not query or len(query) < 3:
            return jsonify({
                "error": "Query must be at least 3 characters"
            }), 400
        
        if len(query) > 5000:
            return jsonify({
                "error": "Query exceeds maximum length (5000 characters)"
            }), 400
        
        # Validate k
        try:
            top_k = int(top_k)
            if top_k < 5 or top_k > 10:
                top_k = 10
        except (ValueError, TypeError):
            top_k = 10
        
        logger.info(f"📝 New recommendation request")
        logger.info(f"   ├─ Query: {query[:50]}...")
        logger.info(f"   └─ Requested k: {top_k}")
        
        # Get recommendations
        recommendations = recommender.get_recommendations(query, k=top_k)
        
        # Build response
        response = {
            "query": query,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Generated {len(recommendations)} recommendations")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Error in /recommend: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/batch_recommend', methods=['POST'])
def batch_recommendations():
    """
    Get recommendations for multiple queries
    
    Request JSON:
    {
        "queries": ["Query 1", "Query 2", ...],
        "top_k": 10
    }
    """
    try:
        if recommender is None:
            return jsonify({"error": "Recommender not initialized"}), 503
        
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({"error": "Missing 'queries' field"}), 400
        
        queries = data.get('queries', [])
        top_k = data.get('top_k', 10)
        
        if not isinstance(queries, list):
            return jsonify({"error": "'queries' must be a list"}), 400
        
        if len(queries) == 0:
            return jsonify({"error": "Queries list is empty"}), 400
        
        if len(queries) > 100:
            return jsonify({"error": "Maximum 100 queries allowed"}), 400
        
        logger.info(f"📝 Batch request for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            try:
                recommendations = recommender.get_recommendations(query, k=top_k)
                results.append({
                    'query': query,
                    'recommendations': recommendations
                })
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        
        logger.info(f"✅ Processed {len(results)} queries in batch")
        
        return jsonify({
            "results": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Batch error: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("\n" + "="*70)
    logger.info("V2.0 ASSESSMENT RECOMMENDATION API SERVER")
    logger.info("="*70)
    
    # Initialize recommender
    if initialize_recommender():
        logger.info("\n🚀 Starting Flask server...")
        logger.info("   ├─ Host: 0.0.0.0")
        logger.info("   ├─ Port: 5000")
        logger.info("   ├─ Debug: True")
        logger.info("   └─ URL: http://127.0.0.1:5000")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
    else:
        logger.error("❌ Cannot start server without recommender")
