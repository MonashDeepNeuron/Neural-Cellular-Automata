This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

[pnpm](https://pnpm.io/) is a faster, more space efficient alternative to [npm](https://www.npmjs.com/). Both managers follow almost identical syntax and pull from the same package repositories by default. If you don't already have pnpm installed, you can do so with this command:

```bash
npm install -g pnpm
```

Install the dependencies for this project with this command:

```bash
pnpm install
```

Then, run the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Linting & Type Checking

[Biome](https://biome.dev) performs linting and type checking is done with [TypeScript](https://www.typescriptlang.org/). It is recommended to open and work on this project folder in [VSCode](https://code.visualstudio.com/) in its own workspace (rather than opening the root folder). This will ensure linting, and TypeScript type checking is works correctly.

You will need to install the `Biome` extension for VSCode. Enabling Biome as a formatter and the `"Format on Save"` option in VSCode settings will make linting on the fly very smooth.

If you ever need to run the linter manually, you can do so with the following command from the root of the repository:

```bash
pnpm run lint
```
