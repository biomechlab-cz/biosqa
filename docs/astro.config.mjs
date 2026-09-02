// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import mdx from '@astrojs/mdx';

// GitHub Pages PROJECT site: served under the /biosqa/ subpath at biomechlab-cz.github.io/biosqa/.
// `base` must prefix every internal link/asset (use the `withBase()` helper, never root-absolute paths).
export default defineConfig({
  site: 'https://biomechlab-cz.github.io',
  base: '/biosqa/',
  trailingSlash: 'always',
  integrations: [mdx(), sitemap()],
  build: { format: 'directory' },
});
