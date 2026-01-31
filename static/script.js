/**
 * Email Shield - Frontend JavaScript
 * ====================================
 * Handles email analysis, stats loading, and UI interactions
 */

// API Base URL
const API_BASE = '';

// DOM Elements
let emailInput, analyzeBtn, clearBtn, resultPanel;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    emailInput = document.getElementById('emailInput');
    emailInput = document.getElementById('emailInput');
    analyzeBtn = document.getElementById('analyzeBtn');
    clearBtn = document.getElementById('clearBtn');
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
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            emailInput.value = '';
            emailInput.focus();
            resultPanel.style.display = 'none'; // Hide results if cleared
        });
    }

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

    // Artificial delay for "Cyberpunk" scanning effect (5 seconds)
    const delay = ms => new Promise(res => setTimeout(res, ms));

    try {
        const [_, response] = await Promise.all([
            delay(5000), // Wait at least 5 seconds
            fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email_text: emailText })
            })
        ]);

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
    // Set icon based on classification
    const resultIcon = document.getElementById('resultIcon');
    const icons = {
        'Legitimate': 'check_circle',
        'Spam': 'warning',
        'Phishing': 'error'
    };
    resultIcon.textContent = icons[prediction] || 'help_outline';

    // Set color class for icon
    resultIcon.className = 'material-icons-outlined result-icon ' + prediction.toLowerCase();

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
        'Legitimate': 'Safe to proceed. Validated by standard protocols.',
        'Spam': 'Marked as Spam. Contains promotional or unsolicited content.',
        'Phishing': 'CRITICAL THREAT. Do not open links or download attachments.'
    };
    const messageEl = document.getElementById('resultMessage');
    if (data.key_signal) {
        messageEl.innerHTML = `<strong>${messages[prediction]}</strong><br><span class="signal-text">Signal: ${escapeHtml(data.key_signal)}</span>`;
    } else {
        messageEl.textContent = messages[prediction];
    }

    // Animate result panel
    resultPanel.style.display = 'block';
    resultPanel.style.animation = 'none';
    resultPanel.offsetHeight; // Trigger reflow
    resultPanel.style.animation = 'fadeIn 0.3s ease-out';

    document.getElementById('resultTitle').textContent = 'Analysis Report';
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

    if (totalEl) totalEl.textContent = stats.total_scans;
    if (legitEl) legitEl.textContent = stats.legitimate;
    if (spamEl) spamEl.textContent = stats.spam;
    if (phishEl) phishEl.textContent = stats.phishing;
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
                <span class="material-icons-outlined empty-icon">inbox</span>
                <p>No recent analysis data.</p>
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
 * Show notification toast (Minimal White)
 */
function showNotification(message, type = 'info') {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;

    const iconMap = {
        'error': 'error_outline',
        'warning': 'warning_amber',
        'success': 'check_circle',
        'info': 'info'
    };

    notification.innerHTML = `
        <span class="material-icons-outlined notification-icon">${iconMap[type] || 'info'}</span>
        <span class="notification-message">${message}</span>
    `;

    // Minimized Styles
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 1.5rem;
                right: 1.5rem;
                padding: 1rem 1.5rem;
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                z-index: 1000;
                animation: slideIn 0.3s ease-out;
                font-size: 0.95rem;
                color: #1E3A8A;
            }
            .notification-error { border-left: 4px solid #DC2626; }
            .notification-warning { border-left: 4px solid #D97706; }
            .notification-success { border-left: 4px solid #16A34A; }
            .notification .material-icons-outlined { font-size: 20px; color: #64748B; }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-10px)';
        notification.style.transition = 'all 0.3s ease';
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
