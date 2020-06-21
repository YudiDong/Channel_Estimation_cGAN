function [pilots] = uniformPilotsGen(num_pilots)
%this function is from https://github.com/YuZhang-GitHub/1-Bit-ADCs

pilots = cell(1, length(num_pilots));
for ii = 1:length(num_pilots)
    pilot_angles = linspace(0, pi/2, num_pilots(ii));
    pilots{1, ii} = exp(1j*pilot_angles.');
end

end

