document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const searchInput = document.getElementById('search-input');
    const iframe = document.getElementById('content-frame');
    const navLinks = document.querySelectorAll('.nav-links a');

    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') document.body.classList.add('dark');

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark');
        const currentTheme = document.body.classList.contains('dark') ? 'dark' : 'light';
        localStorage.setItem('theme', currentTheme);
        syncIframeTheme();
    });

    function syncIframeTheme() {
        if (iframe && iframe.contentDocument && iframe.contentDocument.body) {
            if (document.body.classList.contains('dark')) {
                iframe.contentDocument.body.classList.add('dark');
            } else {
                iframe.contentDocument.body.classList.remove('dark');
            }
            

            if (window.Prism) {
                window.Prism.highlightAllUnder(iframe.contentDocument.body);
            }

            bindSubPageEvents(iframe.contentDocument);
        }
    }
    if (iframe) iframe.addEventListener('load', syncIframeTheme);

    document.querySelectorAll('.category-title').forEach(title => {
        title.addEventListener('click', () => title.parentElement.classList.toggle('collapsed'));
    });

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
        });
    });

    searchInput.addEventListener('input', (e) => {
        const keyword = e.target.value.toLowerCase().trim();
        document.querySelectorAll('.category-item').forEach(category => {
            let hasVisibleLink = false;
            category.querySelectorAll('.nav-links a').forEach(link => {
                const linkText = link.textContent.toLowerCase();
                const keywords = link.getAttribute('data-keywords') ? link.getAttribute('data-keywords').toLowerCase() : '';
                if (linkText.includes(keyword) || keywords.includes(keyword)) {
                    link.parentElement.style.display = 'block';
                    hasVisibleLink = true;
                } else {
                    link.parentElement.style.display = 'none';
                }
            });
            if (hasVisibleLink) {
                category.style.display = 'block';
                category.classList.remove('collapsed');
            } else {
                category.style.display = 'none';
            }
        });
    });

    function bindSubPageEvents(subDoc) {
        subDoc.querySelectorAll('.btn-copy').forEach(btn => {
            if (btn.getAttribute('data-bound')) return;
            btn.setAttribute('data-bound', 'true');
            btn.addEventListener('click', () => {
                const code = btn.nextElementSibling.textContent;
                navigator.clipboard.writeText(code).then(() => {
                    btn.textContent = '已复制';
                    setTimeout(() => btn.textContent = '复制', 2000);
                });
            });
        });
    }
});
