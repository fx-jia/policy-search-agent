import type { Config } from 'jest';

const config: Config = {
  testEnvironment: 'jsdom',
  transform: {
    // Local wrapper around esbuild-jest that also defines import.meta.env;
    // see jest-esbuild-transformer.cjs
    '^.+\\.(ts|tsx|js|jsx)$': '<rootDir>/jest-esbuild-transformer.cjs',
  },
  // Exercise real react-markdown/remark security behavior through their ESM graph.
  transformIgnorePatterns: [
    '/node_modules/(?!(react-markdown|remark-[^/]+|rehype-[^/]+|unified|vfile[^/]*|micromark[^/]*|mdast[^/]*|hast[^/]*|unist[^/]*|bail|is-plain-obj|trough|devlop|property-information|space-separated-tokens|comma-separated-tokens|decode-named-character-reference|character-entities[^/]*|ccount|escape-string-regexp|trim-lines|zwitch|longest-streak|markdown-table|estree-util-is-identifier-name|html-url-attributes|is-alphabetical|is-decimal|is-alphanumerical)/)',
  ],
  moduleNameMapper: {
    // Drags the app shell (routes/react-router) into jsdom; see __mocks__
    '^@/components/layout-recognize-form-field$':
      '<rootDir>/__mocks__/layout-recognize-form-field.js',
    '^@/(.*)$': '<rootDir>/src/$1',
    '^human-id$': '<rootDir>/__mocks__/human-id.js',
    '\\.(css|less|scss|sass)$': '<rootDir>/__mocks__/styleMock.js',
    '\\.(jpg|jpeg|png|gif|svg|webp)$': '<rootDir>/__mocks__/fileMock.js',
  },
  setupFilesAfterEnv: ['<rootDir>/jest-setup.ts'],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx,js,jsx}',
    '!src/.umi/**',
    '!src/.umi-test/**',
    '!src/.umi-production/**',
    '!**/*.d.ts',
    '!coverage/**',
    '!dist/**',
    '!config/**',
    '!mock/**',
  ],
  coverageThreshold: {
    global: {
      lines: 1,
    },
  },
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
};

export default config;
