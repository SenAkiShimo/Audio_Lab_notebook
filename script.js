document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const searchInput = document.getElementById('search-input');
    const iframe = document.getElementById('content-frame');
    const navLinks = document.querySelectorAll('.nav-links a');

    // 1. 初始化与同步暗黑模式
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
            // 给子页面的复制按钮绑定点击事件
            bindSubPageEvents(iframe.contentDocument);
        }
    }
    if (iframe) iframe.addEventListener('load', syncIframeTheme);

    // 2. 大类折叠与高亮切换
    document.querySelectorAll('.category-title').forEach(title => {
        title.addEventListener('click', () => title.parentElement.classList.toggle('collapsed'));
    });

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            navLinks.forEach(l => l.classList.remove('active'));
            e.target.classList.add('active');
        });
    });

    // 3. 🔍 核心功能：跨网页全局模糊搜索
    searchInput.addEventListener('input', (e) => {
        const keyword = e.target.value.toLowerCase().trim();

        document.querySelectorAll('.category-item').forEach(category => {
            let hasVisibleLink = false;
            const linksInCat = category.querySelectorAll('.nav-links a');

            linksInCat.forEach(link => {
                const linkText = link.textContent.toLowerCase();
                const keywords = link.getAttribute('data-keywords').toLowerCase();

                // 只要网页名或者 data-keywords 里包含搜索词，就显示
                if (linkText.includes(keyword) || keywords.includes(keyword)) {
                    link.parentElement.style.display = 'block';
                    hasVisibleLink = true;
                } else {
                    link.parentElement.style.display = 'none';
                }
            });

            // 如果这个大类下所有小类都不符合，就把整个大类隐藏
            if (hasVisibleLink) {
                category.style.display = 'block';
                category.classList.remove('collapsed'); // 搜索时自动展开大类
            } else {
                category.style.display = 'none';
            }
        });
    });

    // 4. 📋 扩展功能：为子网页注入“一键复制代码”逻辑
    function bindSubPageEvents(subDoc) {
        subDoc.querySelectorAll('.btn-copy').forEach(btn => {
            // 防止重复绑定
            if (btn.getAttribute('data-bound')) return;
            btn.setAttribute('data-bound', 'true');

            btn.addEventListener('click', () => {
                const code = btn.nextElementSibling.textContent;
                navigator.clipboard.writeText(code).then(() => {
                    btn.textContent = '✅ 已复制';
                    setTimeout(() => btn.textContent = '📋 复制', 2000);
                });
            });
        });
    }
});