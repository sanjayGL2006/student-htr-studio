/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy: "#0b1220",
        amber: "#f5a524",
        teal: "#14b8a6",
      },
    },
  },
  plugins: [],
};
