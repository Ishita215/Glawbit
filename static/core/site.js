(function () {
    const root = document.documentElement;
    const savedTheme = localStorage.getItem('glawbit-theme') || 'light';

    function applyTheme(theme) {
        const isDark = theme === 'dark';
        root.dataset.theme = isDark ? 'dark' : 'light';
        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            const label = button.querySelector('.theme-label');
            const icon = button.querySelector('.theme-icon');
            if (label) label.textContent = isDark ? 'Light mode' : 'Dark mode';
            if (icon) icon.textContent = isDark ? '☀' : '☾';
            button.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
            button.setAttribute('aria-pressed', String(isDark));
        });
    }

    applyTheme(savedTheme);
    document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
        button.addEventListener('click', () => {
            const nextTheme = root.dataset.theme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('glawbit-theme', nextTheme);
            applyTheme(nextTheme);
        });
    });
}());
