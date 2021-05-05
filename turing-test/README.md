This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://github.com/vercel/next.js/tree/canary/packages/create-next-app).

## Getting Started

First, install all dependencies with `npm install`. You also need MongoDB.

To run a development server, first start a mongodb instance. For example,
you can:

```bash
mkdir db
mongod --dbpath=./db
```

This will run Mongo on the foreground. Thus, open up a new terminal and run:
```bash
npx next dev
```

Open [http://localhost:3000](http://localhost:3000) and you should see the existing user study.

## Code

The current flow is as follows:

- `pages/index.js` creates a new session (also see `pages/api/new-session.js`) and redirects to
- `pages/instructions.js` shows a consent form and an attention check (needed for Mechanical Turk), then redirects to
- `pages/td-instructions.js` shows instructions for the actual task, then redirects to
- `pages/td-exercises.js` shows a series of exercises, recording their responses. Then it redirects to
- `pages/td-post-test.js` shows a series of exercises, recording their responses. Then it redirects to
- `pages/end.js` shows a form to just collect demographic data about the participant, and concludes.

Custom React components are put under `lib/components`. When importing them from pages, we might
need to use a dynamic import (see existing examples) because Next pre-renders everything it can
server-side, but some components can't be server-side rendered.

## Adapting

This is roughly the list of changes we need to make to adapt this app for the turing test:

- Change `pages/instructions.js` to reflect the new task
- Change `pages/td-instructions.js` to reflect the new task
- Change `pages/api/curriculum.js` to instead load the step-by-step solutions from a JSON file (you can just `require` the JSON directly!), and serve those one-by-one, or tell the client that they ended.
- Change `pages/td-exercises.js` to do the actual "Turing Test". Here there are two designs we can explore: we can either show 4 solutions in a random order and then ask people to pick the two that they think were human-written, or we can ask them to rank all 4. The first is easier on the humans and the task is more clear; the second gives us more information. I think both are valid. In any case, we'll likely need a new component for people to make the choice, and we can put that under `lib/components`.
- Make sure answers are being saved. This requires either giving exactly what `pages/api/save-answers.js` already accepts, or modify that as well.
- Finally, the survey at the end can remain unchanged. That's all!