// Theme Management System

class ThemeManager {
    constructor() {
        this.themes = ['dark', 'light'];
        this.currentTheme = this.getStoredTheme() || this.getSystemTheme();
        this.init();
    }

    init() {
        // Apply initial theme
        this.applyTheme(this.currentTheme);
        
        // Setup theme toggle buttons
        this.setupThemeToggles();
        
        // Listen for system theme changes
        this.watchSystemTheme();
        
        // Add keyboard shortcut (Ctrl/Cmd + Shift + T)
        this.setupKeyboardShortcut();
    }

    getStoredTheme() {
        return localStorage.getItem('theme');
    }

    setStoredTheme(theme) {
        localStorage.setItem('theme', theme);
    }

    getSystemTheme() {
        return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    applyTheme(theme) {
        // Validate theme
        if (!this.themes.includes(theme)) {
            theme = 'dark';
        }

        // Apply theme to document
        document.documentElement.setAttribute('data-theme', theme);
        this.currentTheme = theme;
        this.setStoredTheme(theme);

        // Update all theme toggle buttons
        this.updateThemeToggles();

        // Dispatch theme change event
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));

        // Update meta theme-color
        const metaThemeColor = document.querySelector('meta[name="theme-color"]');
        if (metaThemeColor) {
            metaThemeColor.content = theme === 'dark' ? '#0a0a0a' : '#ffffff';
        }
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(newTheme);
        
        // Add animation effect
        this.addTransitionEffect();
    }

    setupThemeToggles() {
        // Find all theme toggle buttons
        const toggleButtons = document.querySelectorAll('.theme-toggle, [data-theme-toggle]');
        
        toggleButtons.forEach(button => {
            // Remove existing listeners to prevent duplicates
            button.replaceWith(button.cloneNode(true));
        });

        // Re-query and add listeners
        document.querySelectorAll('.theme-toggle, [data-theme-toggle]').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleTheme();
            });
        });
    }

    updateThemeToggles() {
        const toggleButtons = document.querySelectorAll('.theme-toggle, [data-theme-toggle]');
        
        toggleButtons.forEach(button => {
            const icon = button.querySelector('i');
            if (icon) {
                icon.className = this.currentTheme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
            }
            
            // Update aria-label for accessibility
            button.setAttribute('aria-label', `Switch to ${this.currentTheme === 'dark' ? 'light' : 'dark'} mode`);
        });
    }

    watchSystemTheme() {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: light)');
        
        mediaQuery.addEventListener('change', (e) => {
            // Only auto-switch if user hasn't manually set a preference
            if (!this.getStoredTheme()) {
                this.applyTheme(e.matches ? 'light' : 'dark');
            }
        });
    }

    setupKeyboardShortcut() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Shift + T
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                this.toggleTheme();
            }
        });
    }

    addTransitionEffect() {
        // Create overlay for smooth transition
        const overlay = document.createElement('div');
        overlay.className = 'theme-transition-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: ${this.currentTheme === 'dark' ? 'rgba(0,0,0,0)' : 'rgba(255,255,255,0)'};
            pointer-events: none;
            z-index: 9999;
            transition: background 0.3s ease;
        `;
        
        document.body.appendChild(overlay);
        
        // Trigger animation
        requestAnimationFrame(() => {
            overlay.style.background = this.currentTheme === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.2)';
        });
        
        // Remove overlay after animation
        setTimeout(() => {
            overlay.remove();
        }, 300);
    }

    // Helper method to get current theme
    getTheme() {
        return this.currentTheme;
    }

    // Helper method to check if dark mode
    isDarkMode() {
        return this.currentTheme === 'dark';
    }

    // Helper method to set specific theme
    setTheme(theme) {
        if (this.themes.includes(theme)) {
            this.applyTheme(theme);
        }
    }
}

// Initialize theme manager when DOM is ready
let themeManager;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        themeManager = new ThemeManager();
    });
} else {
    themeManager = new ThemeManager();
}

// Export for use in other scripts
window.ThemeManager = ThemeManager;
window.themeManager = themeManager;