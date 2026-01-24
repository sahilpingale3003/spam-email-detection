/**
 * Email Shield - Frontend JavaScript
 * ====================================
 * Handles email analysis, stats loading, and UI interactions
 */

// API Base URL
const API_BASE = '';

// DOM Elements
let emailInput, analyzeBtn, resultPanel;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    emailInput = document.getElementById('emailInput');
    analyzeBtn = document.getElementById('analyzeBtn');
    resultPanel = document.getElementById('resultPanel');

    // Initialize dashboard if on index page
    if (analyzeBtn) {
        initializeDashboard();
    }

    // Load stats
    loadStats();
});

/**
 * Initialize dashboard functionality
 */
function initializeDashboard() {
    analyzeBtn.addEventListener('click', analyzeEmail);

    // Allow Ctrl+Enter to submit
    emailInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeEmail();
        }
    });
}

/**
 * Analyze email text
 */
async function analyzeEmail() {
    const emailText = emailInput.value.trim();

    if (!emailText) {
        showNotification('Please enter email content to analyze', 'warning');
        emailInput.focus();
        return;
    }

    // Show loading state
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email_text: emailText })
        });

        const data = await response.json();

        if (data.success) {
            showResult(data);
            loadStats(); // Refresh stats
        } else {
            showNotification(data.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Error analyzing email:', error);
        showNotification('Failed to connect to server. Make sure the server is running.', 'error');
    } finally {
        setLoading(false);
    }
}

/**
 * Show analysis result
 */
function showResult(data) {
    const { prediction, confidence, threat_level } = data;

    // Update result panel
    resultPanel.style.display = 'block';

    // Set icon based on classification
    const resultIcon = document.getElementById('resultIcon');
    const icons = {
        'Legitimate': '✅',
        'Spam': '⚠️',
        'Phishing': '🚨'
    };
    resultIcon.textContent = icons[prediction] || '🔍';

    // Set classification badge
    const badge = document.getElementById('classificationBadge');
    badge.textContent = prediction;
    badge.className = 'classification-badge ' + prediction.toLowerCase();

    // Set confidence
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceFill = document.getElementById('confidenceFill');
    confidenceValue.textContent = confidence.toFixed(1) + '%';
    confidenceFill.style.width = confidence + '%';
    confidenceFill.className = 'confidence-fill ' + threat_level;

    // Set message
    const messages = {
        'Legitimate': 'This email appears to be safe and legitimate. No suspicious patterns detected.',
        'Spam': 'This email shows characteristics of spam. Be cautious of unsolicited offers or promotions.',
        'Phishing': 'WARNING: This email shows signs of phishing! Do not click any links or provide personal information.'
    };
    document.getElementById('resultMessage').textContent = messages[prediction];

    // Animate result panel
    resultPanel.style.animation = 'none';
    resultPanel.offsetHeight; // Trigger reflow
    resultPanel.style.animation = 'slideUp 0.4s ease-out';

    // Update result title
    document.getElementById('resultTitle').textContent = 'Analysis Complete';
}

/**
 * Load dashboard statistics
 */
async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        if (data.success) {
            updateStats(data.stats);
            updateRecentList(data.stats.recent_scans);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

/**
 * Update stats cards
 */
function updateStats(stats) {
    const totalEl = document.getElementById('totalScans');
    const legitEl = document.getElementById('legitimateCount');
    const spamEl = document.getElementById('spamCount');
    const phishEl = document.getElementById('phishingCount');

    if (totalEl) animateNumber(totalEl, stats.total_scans);
    if (legitEl) animateNumber(legitEl, stats.legitimate);
    if (spamEl) animateNumber(spamEl, stats.spam);
    if (phishEl) animateNumber(phishEl, stats.phishing);
}

/**
 * Animate number counting
 */
function animateNumber(element, target) {
    const current = parseInt(element.textContent) || 0;
    const duration = 500;
    const steps = 20;
    const increment = (target - current) / steps;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        element.textContent = Math.round(current + increment * step);

        if (step >= steps) {
            element.textContent = target;
            clearInterval(timer);
        }
    }, duration / steps);
}

/**
 * Update recent scans list
 */
function updateRecentList(recentScans) {
    const recentList = document.getElementById('recentList');
    if (!recentList) return;

    if (!recentScans || recentScans.length === 0) {
        recentList.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📭</span>
                <p>No scans yet. Analyze an email to get started!</p>
            </div>
        `;
        return;
    }

    recentList.innerHTML = recentScans.map(scan => `
        <div class="recent-item">
            <span class="recent-preview">${escapeHtml(scan.email_preview)}</span>
            <span class="recent-badge badge-${getColor(scan.classification)}">${scan.classification}</span>
            <span class="recent-confidence">${scan.confidence.toFixed(0)}%</span>
        </div>
    `).join('');
}

/**
 * Get color class for classification
 */
function getColor(classification) {
    const colors = {
        'Legitimate': 'green',
        'Spam': 'yellow',
        'Phishing': 'red'
    };
    return colors[classification] || 'green';
}

/**
 * Set loading state
 */
function setLoading(isLoading) {
    if (!analyzeBtn) return;

    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnIcon = analyzeBtn.querySelector('.btn-icon');
    const btnLoading = analyzeBtn.querySelector('.btn-loading');

    if (isLoading) {
        analyzeBtn.disabled = true;
        if (btnText) btnText.style.display = 'none';
        if (btnIcon) btnIcon.style.display = 'none';
        if (btnLoading) btnLoading.style.display = 'inline-flex';
    } else {
        analyzeBtn.disabled = false;
        if (btnText) btnText.style.display = 'inline';
        if (btnIcon) btnIcon.style.display = 'inline';
        if (btnLoading) btnLoading.style.display = 'none';
    }
}

/**
 * Show notification toast
 */
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span class="notification-icon">${type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}</span>
        <span class="notification-message">${message}</span>
    `;

    // Add styles if not already added
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 1.5rem;
                right: 1.5rem;
                padding: 1rem 1.5rem;
                background: rgba(0, 0, 0, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                z-index: 1000;
                animation: slideIn 0.3s ease-out;
                backdrop-filter: blur(10px);
            }
            .notification-error { border-color: #ef4444; }
            .notification-warning { border-color: #f59e0b; }
            .notification-success { border-color: #10b981; }
            @keyframes slideIn {
                from { opacity: 0; transform: translateX(100px); }
                to { opacity: 1; transform: translateX(0); }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideIn 0.3s ease-out reverse';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
