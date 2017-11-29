function projections = projectDataOnStepFunction(x, edges)

projections = zeros(length(edges),length(x));

p = 0;

x = x';

binWidth = edges(2)-edges(1);
edges = edges-binWidth/2;
visited_points = zeros(size(x));

for i = 1:length(edges)
    
    points_within_interval = x >= edges(i) & x < edges(i) + binWidth;
    projections(i,points_within_interval) = 1/binWidth;
    visited_points(points_within_interval) = 1;
    
end

projections(1,~points_within_interval) = 1e-5;

end