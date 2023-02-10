clear
close all
P = [0,5; 1,4; 2,3; 3,2; 4,1; 5,0];
theta = [-pi:0.01:pi];
dists = zeros(length(P), length(theta));

figure
hold on
for i=1:length(P)
    for j=1:length(theta)
        dists(i,j) = P(i,1)*cos(theta(j)) + P(i,2)*sin(theta(j));        
    end
    plot(theta, dists(i,:))
end
xlabel("Theta [rad]")
ylabel("Distance")