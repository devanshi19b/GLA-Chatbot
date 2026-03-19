/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./pages/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}",
    "./styles/**/*.{js,jsx,ts,tsx,css}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          ink: "#122033",
          gold: "#c99b2b",
          mist: "#eef4ff",
          blush: "#fff7eb",
        },
      },
      boxShadow: {
        panel: "0 24px 80px rgba(18, 32, 51, 0.12)",
      },
    },
  },
  plugins: [],
};
