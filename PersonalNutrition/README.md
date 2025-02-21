# Personal Nutrition

AI-powered nutrition recommendation system that provides personalized meal plans and dietary advice.

## Features

- Personalized meal recommendations
- Dietary restriction handling
- Nutritional analysis
- Goal-based meal planning
- Integration with health tracking

## Tech Stack

- Frontend: Next.js, TailwindCSS
- Backend: Node.js, Express
- AI: OpenAI API
- Database: PostgreSQL with Drizzle ORM
- Deployment: Vercel

## Project Structure

- `client/` - Next.js frontend application
- `server/` - Express backend API
- `db/` - Database schemas and migrations

## Project Status

Currently in beta deployment with core recommendation features implemented.

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Configure API keys and database connection
```

3. Run development server:
```bash
npm run dev
```

## Documentation

- Frontend pages and components in `client/`
- Backend API endpoints in `server/`
- Database models and migrations in `db/`
