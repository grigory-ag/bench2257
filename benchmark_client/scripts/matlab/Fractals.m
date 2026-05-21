width = 4000;
height = 4000;
maxIter = 1000;

tStart = tic;

mandelbrotCounts = mandelbrotSet(width, height, maxIter);
juliaCounts = juliaSet(width, height, maxIter, -0.7 + 0.27015i);
burningShipCounts = burningShipSet(width, height, maxIter);

elapsed = toc(tStart);
ms = elapsed * 1000;
fprintf('{"execution_time_ms": %.2f}\n', ms);

function counts = mandelbrotSet(width, height, maxIter)
    x = linspace(-2.5, 1.5, width);
    y = linspace(-1.5, 1.5, height);
    [X, Y] = meshgrid(x, y);
    C = X + 1i * Y;
    Z = zeros(height, width);
    counts = zeros(height, width, 'uint16');
    active = true(height, width);

    for iter = 1:maxIter
        Z(active) = Z(active).^2 + C(active);
        escaped = active & (abs(Z) > 2);
        counts(escaped) = iter;
        active(escaped) = false;
        if ~any(active(:))
            break;
        end
    end
    counts(active) = maxIter;
end

function counts = juliaSet(width, height, maxIter, c)
    x = linspace(-1.5, 1.5, width);
    y = linspace(-1.5, 1.5, height);
    [X, Y] = meshgrid(x, y);
    Z = X + 1i * Y;
    counts = zeros(height, width, 'uint16');
    active = true(height, width);

    for iter = 1:maxIter
        Z(active) = Z(active).^2 + c;
        escaped = active & (abs(Z) > 2);
        counts(escaped) = iter;
        active(escaped) = false;
        if ~any(active(:))
            break;
        end
    end
    counts(active) = maxIter;
end

function counts = burningShipSet(width, height, maxIter)
    x = linspace(-2.0, 1.0, width);
    y = linspace(-1.5, 1.5, height);
    [X, Y] = meshgrid(x, y);
    C = X + 1i * Y;
    Z = zeros(height, width);
    counts = zeros(height, width, 'uint16');
    active = true(height, width);

    for iter = 1:maxIter
        Za = abs(real(Z(active))) + 1i * abs(imag(Z(active)));
        Z(active) = Za.^2 + C(active);
        escaped = active & (abs(Z) > 2);
        counts(escaped) = iter;
        active(escaped) = false;
        if ~any(active(:))
            break;
        end
    end
    counts(active) = maxIter;
end
