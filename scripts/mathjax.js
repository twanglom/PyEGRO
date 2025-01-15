

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

// Function to handle equation rendering
function renderMathInElement(elem) {
    if (typeof MathJax !== 'undefined') {
        MathJax.typesetPromise([elem]).catch((err) => console.log('MathJax error:', err));
    }
}

// Handle initial page load
document.addEventListener('DOMContentLoaded', () => {
    renderMathInElement(document.body);
});

// Handle Material for MkDocs instant navigation
document.addEventListener('mdx-navigate', () => {
    setTimeout(() => {
        renderMathInElement(document.body);
    }, 0);
});

// Handle navigation through tabs and links
if (document$) {
    document$.subscribe(() => {
        setTimeout(() => {
            renderMathInElement(document.body);
        }, 0);
    });
}

// Optional: Add a loading state for equations
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = `
        .math-loading { 
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .math-loaded {
            visibility: visible;
            opacity: 1;
        }
    `;
    document.head.appendChild(style);
});








