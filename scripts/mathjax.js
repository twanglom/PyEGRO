// window.MathJax = {
//     tex: {
//       inlineMath: [["\\(", "\\)"]],
//       displayMath: [["\\[", "\\]"], ["$$", "$$"]],
//       processEscapes: true,
//       processEnvironments: true
//     },
//     options: {
//       ignoreHtmlClass: ".*|",
//       processHtmlClass: "arithmatex"
//     },
//     startup: {
//     pageReady: () => {
//       return MathJax.startup.defaultPageReady().then(() => {
//         // Add a hook for instant loading
//         document.addEventListener('DOMContentLoaded', () => {
//           if (typeof MathJax !== 'undefined') {
//             MathJax.typeset();
//           }
//         });
        
//         // Add hook for Material for MkDocs instant loading
//         document$.subscribe(() => {
//           if (typeof MathJax !== 'undefined') {
//             MathJax.typeset();
//           }
//         });
//       });
//     }
//   }
// };



window.MathJax = {
  tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"], ["$$", "$$"]],
      processEscapes: true,
      processEnvironments: true
  },
  options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
  },
  startup: {
      ready: () => {
          MathJax.startup.defaultReady();
      }
  }
};

// Function to render equations
function renderMathJax() {
  if (window.MathJax) {
      window.MathJax.typesetPromise && window.MathJax.typesetPromise();
  }
}

// Listen for navigation events
document.addEventListener('DOMContentLoaded', () => {
  renderMathJax();
});

// For Material for MkDocs instant loading
if (document$) {
  document$.subscribe(() => {
      renderMathJax();
  });
}

// For navigation events
document.addEventListener('navigation', () => {
  renderMathJax();
});

// For tab changes
document.addEventListener('click', (e) => {
  if (e.target.closest('.md-tabs__link') || e.target.closest('.md-nav__link')) {
      setTimeout(renderMathJax, 100);
  }
});