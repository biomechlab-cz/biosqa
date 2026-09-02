// Shared site metadata + the docs sitemap (drives the sidebar, prev/next, and breadcrumbs).
export const SITE = {
  name: 'BioSQA Studio',
  tagline: 'On-device signal-quality overlays for biosignal recordings',
  description:
    'BioSQA Studio opens ECG/PPG/EEG/EDA recordings, detects the modality, and runs a compact neural quality model on the CPU to overlay Q0–Q3 quality segments on the trace — on-device, no cloud.',
  repo: 'https://github.com/biomechlab-cz/biosqa',
  version: 'v0.0.1',
};

// base-aware URL helper (the site is served under /biosqa/).
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${p}`;
}

export type DocLink = { slug: string; title: string };
export type DocSection = { label: string; items: DocLink[] };

// The docs IA (design brief §6). Order here === sidebar order === prev/next order.
export const DOCS_NAV: DocSection[] = [
  {
    label: 'Get started',
    items: [
      { slug: 'getting-started', title: 'Install & run' },
      { slug: 'opening-recordings', title: 'Opening recordings & formats' },
      { slug: 'modality-detection', title: 'Automatic modality detection' },
    ],
  },
  {
    label: 'Concepts',
    items: [
      { slug: 'quality-scale', title: 'The Q0–Q3 quality scale' },
      { slug: 'models', title: 'Models & the model-card contract' },
      { slug: 'runtime-guards', title: 'Runtime guards' },
    ],
  },
  {
    label: 'Using the app',
    items: [
      { slug: 'workspace', title: 'The workspace' },
      { slug: 'segmentation', title: 'Quality segmentation' },
      { slug: 'segment-inspector', title: 'Segment inspector & reshaping' },
      { slug: 'explainability', title: 'Explainability (XAI)' },
      { slug: 'manual-review', title: 'Manual review & corrections' },
      { slug: 'recording-overview', title: 'Recording overview' },
      { slug: 'exporting', title: 'Exporting results' },
      { slug: 'llm-audit', title: 'On-device LLM audit' },
      { slug: 'settings', title: 'Settings reference' },
    ],
  },
  {
    label: 'Help',
    items: [
      { slug: 'faq', title: 'FAQ' },
      { slug: 'troubleshooting', title: 'Troubleshooting' },
      { slug: 'contributing', title: 'Contributing & license' },
    ],
  },
];

export const DOCS_FLAT: DocLink[] = DOCS_NAV.flatMap((s) => s.items);
