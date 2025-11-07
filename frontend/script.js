const API_BASE_URL = 'http://localhost:5000';

// Update character count
document.getElementById('queryInput')?.addEventListener('input', function(e) {
    const count = e.target.value.length;
    document.getElementById('charCount').textContent = count;
});

async function getRecommendations() {
    const query = document.getElementById('queryInput').value.trim();
    const topK = parseInt(document.getElementById('topK').value);
    
    if (!query) {
        showError('Please enter a query or job description');
        return;
    }

    if (query.length < 10) {
        showError('Please provide a more detailed description (at least 10 characters)');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';
    document.getElementById('recommendBtn').disabled = true;
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}. Make sure the API server is running at ${API_BASE_URL}`);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('recommendBtn').disabled = false;
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    if (!data.recommendations || data.recommendations.length === 0) {
        resultsDiv.innerHTML = '<div class="error-message">No recommendations found.</div>';
        return;
    }
    
    let html = `
        <div class="results-header">
            <h2><i class="fas fa-chart-bar"></i> Recommended Assessments</h2>
            <span class="result-count">${data.count} Results</span>
        </div>
        <div class="results-grid">
    `;
    
    data.recommendations.forEach((rec, index) => {
        const score = rec.relevance_score * 100;
        const scoreCategory = score >= 80 ? 'match-high' : score >= 60 ? 'match-medium' : 'match-low';
        const scoreIcon = score >= 80 ? '✓' : score >= 60 ? '≈' : '!';
        
        html += `
            <div class="result-card">
                <div class="result-rank">#${index + 1}</div>
                <h3>${rec.name}</h3>
                <div class="result-metadata">
                    <span class="badge">
                        <i class="fas fa-tag"></i>
                        ${rec.test_type}
                    </span>
                    <span class="badge">
                        <i class="fas fa-clock"></i>
                        ${rec.duration} min
                    </span>
                    <span class="badge ${scoreCategory}">
                        <i class="fas fa-chart-pie"></i>
                        ${scoreIcon} ${score.toFixed(1)}%
                    </span>
                </div>
                <a href="${rec.url}" target="_blank" class="result-link">
                    <i class="fas fa-external-link-alt"></i> View Assessment Details
                </a>
            </div>
        `;
    });
    
    html += '</div>';
    
    resultsDiv.innerHTML = html;
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="error-message">
            <p><i class="fas fa-exclamation-circle"></i> ${message}</p>
        </div>
    `;
}

// Allow Enter key to submit (Ctrl + Enter)
document.getElementById('queryInput')?.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        getRecommendations();
    }
});
