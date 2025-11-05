# Design Document

## Overview

The cyberbullying detection web application will be built as a full-stack web application using Flask as the backend framework to integrate seamlessly with the existing Python ML model. The frontend will use modern HTML5, CSS3, and JavaScript with Chart.js for data visualization. The application will feature a responsive design with cyberbullying awareness-themed visuals and intuitive user experience.

## Architecture

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   ML Model      │
│   (HTML/CSS/JS) │◄──►│   (Flask)       │◄──►│ CyberbullyingDetector │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   File System   │
                       │ (Models/Datasets)│
                       └─────────────────┘
```

### Technology Stack
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Chart.js for interactive data visualization
- **Styling**: Custom CSS with responsive design
- **File Handling**: Python's built-in file operations for dataset management
- **Session Management**: Flask sessions for user authentication
- **ML Integration**: Direct integration with existing CyberbullyingDetector class

## Components and Interfaces

### Backend Components

#### 1. Flask Application Structure
```
app/
├── app.py                 # Main Flask application
├── models/
│   └── cyberbully_detector.py  # Existing ML model (moved)
├── static/
│   ├── css/
│   │   └── style.css     # Main stylesheet
│   ├── js/
│   │   └── main.js       # Frontend JavaScript
│   └── images/
│       └── background.jpg # Cyberbullying awareness background
└── templates/
    ├── base.html         # Base template
    ├── login.html        # Login page
    ├── dashboard.html    # Main dashboard
    ├── predict.html      # Text prediction page
    ├── charts.html       # Data visualization page
    └── dataset.html      # Dataset upload page
```

#### 2. Flask Routes and Endpoints
- `GET /` - Redirect to login or dashboard
- `GET,POST /login` - User authentication
- `GET /logout` - Session termination
- `GET /dashboard` - Main dashboard with overview
- `GET /predict` - Text prediction interface
- `POST /api/predict` - ML prediction API endpoint
- `GET /charts` - Data visualization page
- `GET /api/chart-data` - Chart data API endpoint
- `GET /dataset` - Dataset management page
- `POST /api/upload-dataset` - Dataset upload endpoint
- `POST /api/train-model` - Model training endpoint

#### 3. Session Management
- Simple session-based authentication using Flask sessions
- User credentials stored in a simple dictionary (can be extended to database)
- Session timeout and security measures

### Frontend Components

#### 1. Base Template (base.html)
- Common HTML structure with navigation
- Responsive meta tags and CSS/JS includes
- Navigation bar with active page highlighting
- Cyberbullying-themed background integration

#### 2. Authentication Interface (login.html)
- Clean login form with username/password fields
- Form validation and error message display
- Professional styling with background overlay

#### 3. Dashboard Interface (dashboard.html)
- Welcome message and quick stats overview
- Navigation cards to different sections
- Recent activity summary
- Responsive grid layout

#### 4. Prediction Interface (predict.html)
- Large text area for input
- Predict button with loading states
- Results display with color-coded predictions
- Confidence score visualization
- Clear/reset functionality

#### 5. Charts Interface (charts.html)
- Interactive pie chart for offensive vs non-offensive distribution
- Bar chart for cyberbullying categories/reasons
- Statistics cards with key metrics
- Real-time data updates

#### 6. Dataset Interface (dataset.html)
- File upload drag-and-drop area
- Dataset preview table
- Training progress indicators
- Dataset statistics display

## Data Models

### User Session Data
```python
session = {
    'logged_in': bool,
    'username': str,
    'login_time': datetime
}
```

### Prediction Result Model
```python
prediction_result = {
    'text': str,
    'prediction': str,  # 'Offensive' or 'Non-Offensive'
    'label': int,       # 0 or 1
    'confidence': float, # 0.0 to 1.0
    'timestamp': datetime
}
```

### Dataset Information Model
```python
dataset_info = {
    'filename': str,
    'total_rows': int,
    'offensive_count': int,
    'non_offensive_count': int,
    'upload_time': datetime,
    'columns': list
}
```

### Chart Data Models
```python
# Pie chart data
pie_data = {
    'labels': ['Non-Offensive', 'Offensive'],
    'data': [int, int],
    'colors': ['#28a745', '#dc3545']
}

# Bar chart data for cyberbullying reasons
bar_data = {
    'labels': ['Harassment', 'Hate Speech', 'Threats', 'Discrimination', 'Other'],
    'data': [int, int, int, int, int],
    'colors': ['#ff6384', '#36a2eb', '#cc65fe', '#ffce56', '#ff9f40']
}
```

## Error Handling

### Backend Error Handling
- Try-catch blocks around ML model operations
- File upload validation and error messages
- Session timeout handling
- Model training error recovery
- API endpoint error responses with proper HTTP status codes

### Frontend Error Handling
- Form validation before submission
- AJAX request error handling
- User-friendly error message display
- Loading states and progress indicators
- Graceful degradation for JavaScript failures

### ML Model Error Handling
- Model not trained error handling
- Invalid input text handling
- File format validation for datasets
- Memory management for large datasets

## Testing Strategy

### Backend Testing
- Unit tests for Flask routes using pytest
- ML model integration tests
- File upload functionality tests
- Session management tests
- API endpoint response validation

### Frontend Testing
- Manual testing across different browsers
- Responsive design testing on various screen sizes
- Form validation testing
- Chart rendering and interaction testing
- File upload interface testing

### Integration Testing
- End-to-end user workflow testing
- ML model prediction accuracy validation
- Dataset upload and training workflow
- Authentication flow testing
- Cross-browser compatibility testing

### Performance Testing
- Large dataset upload handling
- Model training performance monitoring
- Concurrent user session handling
- Chart rendering performance with large datasets

## Security Considerations

### Authentication Security
- Session-based authentication with secure session cookies
- Password handling (can be extended to hashing)
- Session timeout implementation
- CSRF protection for forms

### File Upload Security
- File type validation (CSV only)
- File size limits
- Secure file storage location
- Input sanitization for uploaded data

### Data Privacy
- No persistent storage of user predictions (unless explicitly saved)
- Secure handling of uploaded datasets
- Clear data retention policies

## Visual Design Specifications

### Color Scheme
- Primary: #2c3e50 (Dark blue-gray)
- Secondary: #3498db (Blue)
- Success: #28a745 (Green for non-offensive)
- Danger: #dc3545 (Red for offensive)
- Warning: #ffc107 (Yellow for warnings)
- Background overlay: rgba(0, 0, 0, 0.7) for text readability

### Typography
- Primary font: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif
- Headings: Bold, larger sizes with good contrast
- Body text: Regular weight, readable sizes (16px base)

### Background Image Integration
- Cyberbullying awareness themed background image
- Overlay for text readability
- Responsive background sizing
- Consistent across all pages

### Responsive Design Breakpoints
- Mobile: < 768px
- Tablet: 768px - 1024px  
- Desktop: > 1024px

## Performance Optimization

### Frontend Optimization
- Minified CSS and JavaScript
- Optimized background images
- Lazy loading for charts
- Efficient DOM manipulation

### Backend Optimization
- Model caching to avoid reloading
- Efficient file handling for uploads
- Optimized database queries (if extended)
- Response compression

### ML Model Optimization
- Model persistence to avoid retraining
- Batch prediction capabilities
- Memory-efficient text processing
- Vectorizer caching