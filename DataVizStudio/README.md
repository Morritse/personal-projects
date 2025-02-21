# Data Visualization Studio

Interactive data visualization platform for creating and sharing data stories.

## Features

- Interactive visualization creation
- Multiple chart types and layouts
- Data import and transformation
- Sharing and collaboration tools

## Tech Stack

- Frontend: React, D3.js, TailwindCSS
- Backend: Node.js, Express
- Database: PostgreSQL
- Deployment: Vercel

## Project Status

Currently in beta testing phase with core visualization features implemented.

## Project Structure

- `client/` - React frontend application
- `server/` - Express backend API
- `shared/` - Shared types and utilities
- `@shared/` - Common components and helpers
- `attached_assets/` - Static assets and resources

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your configuration
```

3. Run development server:
```bash
npm run dev
```

## Documentation

- Frontend components and visualizations in `client/`
- Backend API and data processing in `server/`
- Shared utilities in `shared/`
