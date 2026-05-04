/**
 * Единый словарь задач для таблиц, графиков и подписей (подключать до inline-скриптов).
 */
(function (global) {
    const ORDER = [
        "Sum of Matrices",
        "Multiply Matrices",
        "Invert Matrices",
        "Random Walk",
        "Fractals",
        "Zombie Apocalypse",
        "Pandemic Spread",
    ];

    const NAMES_RU = {
        "Sum of Matrices": "Сложение матриц",
        "Multiply Matrices": "Умножение матриц",
        "Invert Matrices": "Обращение матриц",
        "Random Walk": "Случайное блуждание",
        "Fractals": "Фракталы",
        "Zombie Apocalypse": "Зомби-апокалипсис",
        "Pandemic Spread": "Пандемия",
    };

    global.ALL_BENCH_TASKS = ORDER.slice();
    global.taskNamesRU = NAMES_RU;

    global.taskTitle = function (taskKey) {
        if (taskKey == null || taskKey === "") {
            return "—";
        }
        return NAMES_RU[taskKey] || String(taskKey);
    };
})(typeof window !== "undefined" ? window : this);
