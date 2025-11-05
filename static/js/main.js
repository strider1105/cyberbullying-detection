// Main JavaScript file for Cyberbullying Detection System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize common functionality
    initializeNavigation();
    initializeAlerts();
    initializeTooltips();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Initialize page-specific functionality
    const currentPage = getCurrentPage();
    switch(currentPage) {
        case 'dashboard':
            initializeDashboard();
            break;
        case 'predict':
            initializePrediction();
            break;
        case 'charts':
            initializeCharts();
            break;
        case 'dataset':
            initializeDataset();
            break;
    }
});

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Add loading state for navigation
            if (!this.classList.contains('active')) {
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            }
        });
    });
}

// Alert auto-dismiss functionality
function initializeAlerts() {
    const alerts = document.querySelectorAll('.alert');
    
    alerts.forEach(alert => {
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-20px)';
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 300);
        }, 5000);
        
        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '<i class="fas fa-times"></i>';
        closeBtn.style.cssText = `
            background: none;
            border: none;
            float: right;
            font-size: 1.2rem;
            cursor: pointer;
            opacity: 0.7;
            margin-left: 1rem;
        `;
        closeBtn.addEventListener('click', () => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-20px)';
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 300);
        });
        alert.appendChild(closeBtn);
    });
}

// Tooltip functionality
function initializeTooltips() {
    const elementsWithTooltips = document.querySelectorAll('[title]');
    
    elementsWithTooltips.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

function showTooltip(e) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.textContent = e.target.getAttribute('title');
    tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        z-index: 10000;
        pointer-events: none;
        white-space: nowrap;
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = e.target.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
    
    // Remove title to prevent default tooltip
    e.target.setAttribute('data-original-title', e.target.getAttribute('title'));
    e.target.removeAttribute('title');
}

function hideTooltip(e) {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) {
        tooltip.remove();
    }
    
    // Restore original title
    const originalTitle = e.target.getAttribute('data-original-title');
    if (originalTitle) {
        e.target.setAttribute('title', originalTitle);
        e.target.removeAttribute('data-original-title');
    }
}

// Smooth scrolling
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Get current page
function getCurrentPage() {
    const path = window.location.pathname;
    if (path.includes('dashboard')) return 'dashboard';
    if (path.includes('predict')) return 'predict';
    if (path.includes('charts')) return 'charts';
    if (path.includes('dataset')) return 'dataset';
    return 'dashboard';
}

// Dashboard specific functionality
function initializeDashboard() {
    // Add hover effects to dashboard cards
    const dashboardCards = document.querySelectorAll('.dashboard-card');
    
    dashboardCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// Prediction specific functionality
function initializePrediction() {
    // Add character counter to text input
    const textInput = document.getElementById('textInput');
    if (textInput) {
        const counter = document.createElement('div');
        counter.style.cssText = `
            text-align: right;
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-top: 0.5rem;
        `;
        textInput.parentNode.appendChild(counter);
        
        function updateCounter() {
            const length = textInput.value.length;
            counter.textContent = `${length} characters`;
            
            if (length > 1000) {
                counter.style.color = '#dc3545';
            } else if (length > 500) {
                counter.style.color = '#ffc107';
            } else {
                counter.style.color = '#7f8c8d';
            }
        }
        
        textInput.addEventListener('input', updateCounter);
        updateCounter();
    }
}

// Charts specific functionality
function initializeCharts() {
    // Add chart refresh functionality
    const refreshBtn = document.querySelector('[onclick="refreshData()"]');
    if (refreshBtn) {
        // Add keyboard shortcut for refresh (Ctrl+R or Cmd+R)
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                refreshData();
            }
        });
    }
}

// Dataset specific functionality
function initializeDataset() {
    // Add file validation
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                validateFile(file);
            }
        });
    }
}

function validateFile(file) {
    const validTypes = ['text/csv', 'application/vnd.ms-excel'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
        showNotification('Please select a valid CSV file.', 'error');
        return false;
    }
    
    if (file.size > maxSize) {
        showNotification('File size must be less than 16MB.', 'error');
        return false;
    }
    
    return true;
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)}"></i>
        ${message}
    `;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        min-width: 300px;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Loading state management
function showLoading(element, text = 'Loading...') {
    const originalContent = element.innerHTML;
    element.setAttribute('data-original-content', originalContent);
    element.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${text}`;
    element.disabled = true;
}

function hideLoading(element) {
    const originalContent = element.getAttribute('data-original-content');
    if (originalContent) {
        element.innerHTML = originalContent;
        element.removeAttribute('data-original-content');
    }
    element.disabled = false;
}

// Form validation
function validateForm(formElement) {
    const requiredFields = formElement.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('error');
            field.style.borderColor = '#dc3545';
            isValid = false;
        } else {
            field.classList.remove('error');
            field.style.borderColor = '#ddd';
        }
    });
    
    return isValid;
}

// API helper functions
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        showNotification('Network error. Please try again.', 'error');
        throw error;
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .error {
        animation: shake 0.5s ease-in-out;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

// Global error handler
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    showNotification('An unexpected error occurred. Please refresh the page.', 'error');
});

// Prevent form resubmission on page refresh
if (window.history.replaceState) {
    window.history.replaceState(null, null, window.location.href);
}