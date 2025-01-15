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
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        // Add a hook for instant loading
        document.addEventListener('DOMContentLoaded', () => {
          if (typeof MathJax !== 'undefined') {
            MathJax.typeset();
          }
        });
        
        // Add hook for Material for MkDocs instant loading
        document$.subscribe(() => {
          if (typeof MathJax !== 'undefined') {
            MathJax.typeset();
          }
        });
      });
    }
  }
};