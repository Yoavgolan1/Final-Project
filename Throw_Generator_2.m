clear all
timesteps = 100;
max_time = 10;
t = linspace (0,max_time,timesteps);
g = 9.81;
max_velocity = 100;

N_trials = 1000;

All_Throws = zeros(timesteps, N_trials*3);
for ii = 1:N_trials
    initial_velocity = rand()*max_velocity;
    initial_azimuth = deg2rad(randi(90));
    initial_elevation = deg2rad(randi(90));
    v0x = initial_velocity*cos(initial_azimuth);
    v0y = initial_velocity*sin(initial_azimuth);
    v0z = initial_velocity*sin(initial_elevation);
    
    x_t = 0 + v0x*t;
    y_t = 0 + v0y*t;
    z_t = 0 + v0z*t + 0.5*(-g)*t.^2;
    
    One_Throw = [x_t', y_t', z_t'];
    All_Throws(:,3*(ii-1)+1) = x_t';
    All_Throws(:,3*(ii-1)+2) = y_t';
    All_Throws(:,3*(ii-1)+3) = z_t';
end

csvwrite('input_trajectories_big.csv',All_Throws)