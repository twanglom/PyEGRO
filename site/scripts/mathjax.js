// This configuration is specifically tuned for ReadTheDocs theme
window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"], ["$", "$"]],
      displayMath: [["\\[", "\\]"], ["$$", "$$"]],
      processEscapes: true,
      processEnvironments: true,
      packages: {'[+]': ['ams', 'noerrors']}
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex|math"
    },
    startup: {
      ready: function() {
        MathJax.startup.defaultReady();
        console.log("MathJax is ready");
        
        // This helps with ReadTheDocs theme
        document.body.classList.add('mathjax-loaded');
      }
    }
  };
  
  // Add special handling for ReadTheDocs theme
  document.addEventListener('DOMContentLoaded', function() {
    document.body.classList.add('mathjax-loading');
    
    // ReadTheDocs theme doesn't always process math blocks correctly
    // This adds the proper classes to ensure MathJax finds them
    setTimeout(function() {
      // Find all math content
      const mathBlocks = document.querySelectorAll('div.math, span.math');
      mathBlocks.forEach(function(block) {
        if (!block.classList.contains('arithmatex')) {
          block.classList.add('arithmatex');
        }
      });
    }, 100);
  });