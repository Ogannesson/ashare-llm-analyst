const overlap = document.querySelector('.overlap');
const len = overlap.textContent.length;

// 将文字分割成单个字符并添加动画
overlap.innerHTML = overlap.textContent.split('')
    .map((char, i) => `<span style="z-index: ${len - i}; display: inline-block; opacity: 0; transform: translateY(20px);">${char}</span>`)
    .join('');

const chars = overlap.querySelectorAll('span');

chars.forEach((char, i) => {
    setTimeout(() => {
        char.style.transition = 'all 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
        char.style.opacity = '1';
        char.style.transform = 'translateY(0)';
    }, i * 100);
});

overlap.addEventListener('mouseenter', () => {
    chars.forEach((char, i) => {
        setTimeout(() => {
            char.style.transition = 'all 0.3s ease';
            char.style.transform = 'translateY(-5px) scale(1.1)';
            char.style.color = '#ff6b6b';
        }, i * 50);
    });
});

overlap.addEventListener('mouseleave', () => {
    chars.forEach((char, i) => {
        setTimeout(() => {
            char.style.transition = 'all 0.3s ease';
            char.style.transform = 'translateY(0) scale(1)';
            char.style.color = '';
        }, i * 30);
    });
});

overlap.addEventListener('click', () => {
    chars.forEach((char, i) => {
        setTimeout(() => {
            char.style.transition = 'all 0.2s ease';
            char.style.transform = 'scale(1.3)';

            setTimeout(() => {
                char.style.transform = 'scale(1)';
            }, 150);
        }, i * 20);
    });
});

setInterval(() => {
    const randomChar = chars[Math.floor(Math.random() * chars.length)];
    randomChar.style.transition = 'all 0.3s ease';
    randomChar.style.textShadow = '0 0 10px currentColor';

    setTimeout(() => {
        randomChar.style.textShadow = '';
    }, 300);
}, 3000);

overlap.style.userSelect = 'none';

// 磁铁吸附效果
overlap.addEventListener('mousemove', (e) => {
    const rect = overlap.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    chars.forEach((char) => {
        const charRect = char.getBoundingClientRect();
        const charX = charRect.left - rect.left + charRect.width / 2;
        const charY = charRect.top - rect.top + charRect.height / 2;

        // 计算鼠标到字符的距离
        const distance = Math.sqrt(Math.pow(mouseX - charX, 2) + Math.pow(mouseY - charY, 2));

        // 磁铁效果范围
        const magnetRange = 80;

        if (distance < magnetRange) {
            // 计算吸附力度（距离越近力度越强）
            const force = (magnetRange - distance) / magnetRange;

            // 计算吸附方向
            const angle = Math.atan2(mouseY - charY, mouseX - charX);
            const pullX = Math.cos(angle) * force * 15;
            const pullY = Math.sin(angle) * force * 15;

            char.style.transition = 'all 0.1s ease-out';
            char.style.transform = `translate(${pullX}px, ${pullY}px) scale(${1 + force * 0.2})`;
            char.style.color = `hsl(${Math.floor(force * 60)}, 70%, 60%)`;
        }
    });
});

// 鼠标离开时重置位置
overlap.addEventListener('mouseleave', () => {
    chars.forEach((char) => {
        char.style.transition = 'all 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
        char.style.transform = 'translate(0, 0) scale(1)';
        char.style.color = '';
    });
});