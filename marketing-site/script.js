const menuButton = document.querySelector('.menu-button');
const navigation = document.querySelector('#primary-nav');

menuButton?.addEventListener('click', () => {
  const open = navigation.classList.toggle('is-open');
  menuButton.setAttribute('aria-expanded', String(open));
});

navigation?.addEventListener('click', () => {
  navigation.classList.remove('is-open');
  menuButton?.setAttribute('aria-expanded', 'false');
});
