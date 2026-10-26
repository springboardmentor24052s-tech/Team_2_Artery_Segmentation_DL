ArteryAI – Frontend

1. Project Overview

ArteryAI is the frontend application for Artery Segmentation and Stenosis Detection Using Advanced Deep Learning Techniques. It provides a user-friendly interface for uploading angiography images, managing cases, viewing processing status, and visualizing segmentation results when available.

The frontend is designed to integrate with a backend API that handles image processing and deep learning model inference.

2. Features

- Dashboard: Displays an overview of cases and project activity.
- Case Management: View and organize patient imaging cases using the available case information.
- Image Upload: Provides an interface for selecting angiography images for processing.
- Processing Status: Displays the available status of image-processing tasks.
- Segmentation Results: Provides a UI for viewing segmentation outputs when available.
- Mock/Demo Data: Supports demonstration using sample data where configured.
- Responsive UI: Designed to work across different screen sizes.
- Reusable Components: Uses reusable UI components to maintain consistency across pages.

3. Technology Stack

- React – Building the user interface
- Vite – Development server and build tool
- Tailwind CSS – Styling and responsive layouts
- React Router – Client-side navigation
- Axios – HTTP requests to backend APIs
- Lucide React – Icons
- JavaScript – Frontend programming language

4. Project Structure

frontend/
├── public/
├── src/
│ ├── components/ # Reusable UI components
│ ├── pages/ # Application pages
│ ├── services/ # API and application services
│ ├── contexts/ # Shared application state
│ ├── mocks/ # Sample data for demo mode
│ ├── App.jsx # Main application component
│ └── main.jsx # Application entry point
├── .env.example # Example environment configuration
├── index.html
├── package.json
├── vite.config.js
└── README.md

The structure above is illustrative. Keep only folders and files that exist in your project.

5. Prerequisites

Install the following before running the application:

- Node.js (a version compatible with the project dependencies)
- npm
- Git

6. Installation

Navigate to the frontend directory:

cd frontend

Install dependencies:

npm install

7. Environment Configuration

Create a local .env file based on .env.example, if the project uses these variables.

Example configuration:

VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK=true
VITE_SUPPORTED_IMAGE_FORMATS=jpg,jpeg,png
VITE_MAX_UPLOAD_SIZE_MB=10

Adjust the values according to your actual implementation and backend configuration. Do not commit private credentials or secrets.

8. Run the Application

Start the Vite development server:

npm run dev

Open the local URL printed in your terminal, usually:

http://localhost:5173

9. Mock and Demo Mode

If mock mode is implemented, the frontend can use sample case data to demonstrate the interface before the backend integration is complete.

The mock mode may be configured through VITE_USE_MOCK, depending on how the application reads that variable.

Mock data is for development and demonstration purposes only. It must not be presented as a real patient result or actual model prediction.

10. Backend Integration

The frontend is intended to communicate with backend APIs for image upload, case management, processing status, and segmentation results.

The API base URL can be configured using VITE_API_BASE_URL if supported by the application.

Backend endpoints and response formats should be confirmed with the backend team. Any integration that is not yet available should be documented as planned rather than implemented.

11. Build the Application

To create a production build:

npm run build

To preview the production build locally, if the preview script is configured:

npm run preview

12. Development Notes

- Keep components reusable and maintain a consistent UI.
- Use responsive layouts and accessible form controls.
- Validate image file types and sizes before upload.
- Handle API errors and loading states clearly.
- Keep mock data separate from actual backend responses.
- Do not display mock segmentation results as real medical findings.
- Test changes before pushing them to the shared repository.

13. Project Status

The frontend is being developed as part of the Infosys Springboard internship project. Features and backend integrations may evolve as development progresses.

This frontend is intended for project development and demonstration. Its interface does not independently provide a clinically validated diagnosis.

⸻

Project: ArteryAI – Artery Segmentation and Stenosis Detection
Module: Frontend
Framework: React + Vite
Styling: Tailwind CSS
