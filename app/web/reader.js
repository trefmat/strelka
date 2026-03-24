function toSafeInt(value, fallback = 0) {
  const n = Number.parseInt(String(value), 10);
  return Number.isNaN(n) ? fallback : n;
}

function parseParams() {
  const params = new URLSearchParams(window.location.search);
  const book = (params.get('book') || '').trim();
  const start = Math.max(0, toSafeInt(params.get('start'), 0));
  const end = Math.max(start, toSafeInt(params.get('end'), start));
  return { book, start, end };
}

function setTextNodeRange(container, text, start, end) {
  container.textContent = '';
  if (!text) {
    return;
  }

  const s = Math.max(0, Math.min(start, text.length));
  const e = Math.max(s, Math.min(end, text.length));

  if (s === e) {
    container.textContent = text;
    return;
  }

  const left = text.slice(0, s);
  const mid = text.slice(s, e);
  const right = text.slice(e);

  if (left) container.appendChild(document.createTextNode(left));
  const mark = document.createElement('mark');
  mark.className = 'reader-hit';
  mark.textContent = mid;
  container.appendChild(mark);
  if (right) container.appendChild(document.createTextNode(right));

  requestAnimationFrame(() => {
    mark.scrollIntoView({ block: 'center', behavior: 'smooth' });
  });
}

async function loadReader() {
  const title = document.getElementById('readerTitle');
  const meta = document.getElementById('readerMeta');
  const textBox = document.getElementById('readerText');
  const { book, start, end } = parseParams();

  if (!book) {
    title.textContent = 'Читалка';
    meta.textContent = 'Книга не выбрана';
    textBox.textContent = 'Передайте параметр ?book=<название_книги>.';
    return;
  }

  title.textContent = `Читалка: ${book}`;
  meta.textContent = `Переход к позиции ${start}-${end}`;
  textBox.textContent = 'Обработка...';

  try {
    const res = await fetch(`/books/content?book=${encodeURIComponent(book)}`);
    if (!res.ok) {
      let detail = 'Не удалось открыть книгу';
      const body = await res.text();
      if (body) {
        try {
          const data = JSON.parse(body);
          detail = data.detail || detail;
        } catch (e) {
          const stripped = body
            .replace(/<[^>]+>/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
          if (stripped) {
            detail = `${detail} (HTTP ${res.status}): ${stripped.slice(0, 240)}`;
          } else {
            detail = `${detail} (HTTP ${res.status})`;
          }
        }
      } else {
        detail = `${detail} (HTTP ${res.status})`;
      }
      textBox.textContent = detail;
      return;
    }

    const text = await res.text();
    setTextNodeRange(textBox, text, start, end);
    meta.textContent = `Длина книги: ${text.length} символов`;
  } catch (err) {
    const msg = (err && err.message) ? err.message : 'network error';
    textBox.textContent = `Не удалось открыть книгу: ${msg}`;
  }
}

document.getElementById('readerBackBtn').addEventListener('click', () => {
  if (window.history.length > 1) {
    window.history.back();
  } else {
    window.location.href = '/';
  }
});

document.getElementById('readerTopBtn').addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

loadReader();
