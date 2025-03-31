// Minimal script to fix navigation issues
document.addEventListener('DOMContentLoaded', function() {
    // This code will run after the basic DOM is loaded
    console.log("Navigation script loaded");
    
    // Function to apply navigation fixes
    function fixNavigation() {
      console.log("Applying navigation fixes");
      
      // Get all top-level navigation items that have children
      const topNavItems = document.querySelectorAll('.md-nav--primary > .md-nav__list > .md-nav__item--nested');
      
      topNavItems.forEach(function(item) {
        // Close all sections by default
        if (!item.classList.contains('md-nav__item--active')) {
          item.classList.remove('md-nav__item--expanded');
          const submenu = item.querySelector('.md-nav');
          if (submenu) {
            submenu.setAttribute('hidden', '');
          }
        }
      });
    }
    
    // Apply fixes after a short delay to ensure the DOM is fully processed
    setTimeout(fixNavigation, 100);
  });
  
  // Add performance boost
  window.addEventListener('load', function() {
    // This runs when the entire page is loaded (images, styles, etc.)
    console.log("Page fully loaded");
    
    // Mark page as loaded to trigger any CSS improvements
    document.body.classList.add('page-loaded');
  });