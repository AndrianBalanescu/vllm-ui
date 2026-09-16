/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./vllm_ui/static/index.html'],
  darkMode: 'class',
  safelist: [
    'max-w-[1700px]', 'min-h-[140px]', 'min-h-[220px]',
    'text-[10px]', 'text-[11px]', 'w-[1px]',
    'p-1.5'
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        brand: {
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
        },
        dark: {
          950: '#06090f',
          900: '#0a0f1d',
          850: '#0f172a',
          800: '#131e36',
          750: '#1a2642',
          700: '#233252',
        }
      }
    }
  },
  plugins: []
}
