(() => {
  'use strict';
  const tasks = {
    grasp: ['Handle position, nearby clearance, hand configuration', 'Plan an approach to the red mug.'],
    sort: ['Which mug is red and which is blue', 'Assign each mug to its color group.'],
    count: ['Two distinct mug instances', 'Return the number of mugs: 2.']
  };
  document.querySelectorAll('[data-task-focus]').forEach(root => {
    const controls = root.querySelector('.uni-task-controls');
    const buttons = Array.from(root.querySelectorAll('[data-task-choice]'));
    buttons.forEach(button => button.addEventListener('click', () => {
      const task = button.dataset.taskChoice;
      if (!tasks[task]) return;
      root.dataset.task = task;
      buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      root.querySelector('[data-task-retain]').textContent = tasks[task][0];
      root.querySelector('[data-task-next]').textContent = tasks[task][1];
    }));
    controls.hidden = false;
  });
})();
