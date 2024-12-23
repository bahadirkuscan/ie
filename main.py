# Import the MPI module from the mpi4py library
import math
import sys

import Classes
from mpi4py import MPI

# Initialize the communicator, which is the default communication group
comm = MPI.COMM_WORLD

# Get the total number of processes in the communicator
n_ranks = comm.Get_size()

# Get the rank (unique ID) of the current process in the communicator
rank = comm.Get_rank()
# Gets absolute coordinates of a cell and returns the rank of its processor and relative coordinates
def get_relative_coordinates(x, y, sub_grid_size):
    x_relative = x % sub_grid_size
    y_relative = y % sub_grid_size
    row = x // sub_grid_size
    col = y // sub_grid_size
    rank = int(row * math.sqrt(n_ranks - 1)) + col + 1
    return rank, x_relative, y_relative

def get_absolute_coordinates(rel_x, rel_y, sub_grid_size, source_rank):
    row = int((source_rank-1) // math.sqrt(n_ranks - 1))
    col = int((source_rank-1) % math.sqrt(n_ranks - 1))
    x_abs = row * sub_grid_size + rel_x
    y_abs = col * sub_grid_size + rel_y
    return x_abs, y_abs

# Gets relative rank of a processor given relative addresses
def get_target_rank_offset(new_x, new_y):
    row_diff = new_x // sub_grid_size
    col_diff = new_y // sub_grid_size
    if col_diff == -1 and int(rank % math.sqrt(n_ranks - 1)) == 1:
        return None
    if col_diff == 1 and int(rank % math.sqrt(n_ranks - 1)) == 0:
        return None
    if row_diff == -1 and rank <= math.sqrt(n_ranks - 1):
        return None
    if row_diff == 1 and rank > n_ranks - math.sqrt(n_ranks):
        return None
    return int(row_diff * math.sqrt(n_ranks - 1) + col_diff)


# Parses a wave input
def parse_units(lines):
    units = [[] for _ in range(n_ranks)]
    for i in range(4):
        line = lines[i]
        line = line.split(":")
        positions = line[1].split(",")
        for j in range(unit_count):
            x, y = map(int, positions[j].split())
            target_rank, x_relative, y_relative = get_relative_coordinates(x, y, sub_grid_size)
            units[target_rank].append((line[0], x_relative, y_relative))
    return units


# Returns the rank of a processor using its row and col number
def get_rank(processor_row, processor_col):
    return processor_row * Classes.Grid.grid_index_limit + processor_col + 1



# Returns the number of targets that can be shot by an imaginary air unit deployed in a specific coordinate
def air_unit_target_count(row, col):
    target_number = 0
    for [x, y] in Classes.AirUnit.attack_pattern:

        target_x, target_y = row + x, col + y
        rank_offset = get_target_rank_offset(target_x, target_y)
        if rank_offset is None:
            continue

        # If the cell under check is in the same sub-grid
        if rank_offset == 0:
            target_unit = grid.units[target_x][target_y]
            # Cannot shoot allies
            if isinstance(target_unit, Classes.AirUnit):
                continue

            # Air units can skip over neutral cells, check the next one
            elif target_unit == ".":
                target_x, target_y = row + 2 * x, col + 2 * y
                rank_offset = get_target_rank_offset(target_x, target_y)
                # Target is not in the grid
                if rank_offset is None:
                    continue
                # If the cell under check is in the same sub-grid
                if rank_offset == 0:
                    target_unit = grid.units[target_x][target_y]
                    if target_unit != "." and not isinstance(target_unit, Classes.AirUnit):
                        target_number += 1
                # If the cell under check is in a different sub-grid
                else:
                    comm.send(("send unit type", target_x % sub_grid_size, target_y % sub_grid_size), dest=rank+rank_offset)
                    # Receive unit type of the cell
                    unit_type = comm.recv(source=rank+rank_offset)
                    if unit_type is not str and unit_type is not Classes.AirUnit:
                        target_number += 1
            else:
                target_number += 1
        # If the cell under check is in a different sub-grid
        else:
            comm.send(("send unit type", target_x % sub_grid_size, target_y % sub_grid_size), dest=rank+rank_offset)
            unit_type = comm.recv(source=rank+rank_offset)
            if unit_type is Classes.AirUnit:
                continue
            # Check the next one, target certainly will not be in the same sub-grid as the air unit
            if unit_type is str:
                target_x, target_y = row + 2 * x, col + 2 * y
                rank_offset = get_target_rank_offset(target_x, target_y)
                # Target is not in the grid
                if rank_offset is None:
                    continue
                comm.send(("send unit type", target_x % sub_grid_size, target_y % sub_grid_size),dest=rank + rank_offset)
                unit_type = comm.recv(source=rank + rank_offset)
                if unit_type is not str and unit_type is not Classes.AirUnit:
                    target_number += 1
            else:
                target_number += 1

    return target_number


# Decides on a movement for every air unit in the sub-grid, returns an array consisting of [x, y, new_x, new_y] arrays where x,y are the old coordinates and new_x,new_y are the new coordinates for an air unit.
def air_unit_movement():
    result_array = []
    # Only look at the middle region
    for row in range(sub_grid_size):
        for col in range(sub_grid_size):
            # For each air unit
            if isinstance(grid.units[row][col], Classes.AirUnit):
                # Get target number for each possible movement
                max_targets = air_unit_target_count(row, col)
                # new x and new y will be represented
                new_x, new_y = row, col
                for [x, y] in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
                    target_x, target_y = row + x, col + y
                    rank_offset = get_target_rank_offset(target_x, target_y)
                    if rank_offset is None:
                        continue
                    if rank_offset == 0:
                        if grid.units[target_x][target_y] == ".":
                            target_number = air_unit_target_count(target_x, target_y)
                            if target_number > max_targets:
                                new_x, new_y = target_x, target_y
                                max_targets = target_number
                    else:
                        comm.send(("send target count", target_x % sub_grid_size, target_y % sub_grid_size), dest=rank+rank_offset)
                        response = comm.recv(source=rank+rank_offset)
                        # Neighbour grids will ask for units
                        while type(response) != int:
                            comm.send(type(grid.units[response[1]][response[2]]), dest=rank+rank_offset)
                            response = comm.recv(source=rank+rank_offset)
                        target_number = response
                        if target_number > max_targets:
                            new_x, new_y = target_x, target_y
                            max_targets = target_number
                result_array.append([row, col, new_x, new_y])
    return result_array








if rank == 0:
    # Initialize
    input_file = open(sys.argv[1], 'r')
    output_file = open(sys.argv[2], "w")
    line = input_file.readline()
    # The size of main grid = N
    main_grid_size, wave_count, unit_count, round_count = map(int, line.split())

    # Calculate size of each grid
    sub_grid_size = int(main_grid_size // math.sqrt(n_ranks - 1))
    Classes.Grid.grid_index_limit = main_grid_size // sub_grid_size

    for i in range(1, n_ranks):
        # Send the sub-grid size to workers and wait for them to initialize sub-grids
        comm.send((main_grid_size,sub_grid_size,wave_count,round_count), i)
    for i in range(1, n_ranks):
        comm.recv(source=i)
    for wave_number in range(wave_count):
        # Get the coordinates of each unit and put them in an array
        line = input_file.readline()
        lines = []
        for i in range(4):
            line = input_file.readline()
            lines.append(line)
        units = parse_units(lines)

        # Send each array to corresponding processor
        for i in range(1, n_ranks):
            comm.send(units[i], i)

        # Wait for all the workers to place the new units
        for rank in range(1, n_ranks):
            comm.recv(source=rank)

        for round_number in range(round_count):
            # MOVEMENT PHASE
            # Start with even-even coords and end with odd-odd coords
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                signal_count = 0
                for processor_row in range(a, Classes.Grid.grid_index_limit, 2):
                    for processor_col in range(b, Classes.Grid.grid_index_limit, 2):
                        comm.send("proceed", get_rank(processor_row, processor_col))
                        signal_count += 1
                for _ in range(signal_count):
                    comm.recv()

            for i in range(1, n_ranks):
                comm.send("finish", i)


            # After finishing calculating phase of movement phase now we need to apply those movements
            # First every worker will process its own queue
            for _ in range(n_ranks - 1):
                comm.recv()
            # Once every worker is finished with their own queue signal all the workers to continue
            for i in range(1, n_ranks):
                comm.send("queue finished", i)

            # After that each worker will process requests from other workers
            for _ in range(n_ranks - 1):
                comm.recv()
            # Once each request is finished the current phase is also finished
            for i in range(1, n_ranks):
                comm.send("phase finished", i)

            # ACTION PHASE
            # Start with even-even coords and end with odd-odd coords
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                signal_count = 0
                for processor_row in range(a, Classes.Grid.grid_index_limit, 2):
                    for processor_col in range(b, Classes.Grid.grid_index_limit, 2):
                        comm.send("proceed", get_rank(processor_row, processor_col))
                        signal_count += 1
                for _ in range(signal_count):
                    comm.recv()

            for i in range(1, n_ranks):
                comm.send("decisions finished",dest= i)

            for i in range(1, n_ranks):
                comm.recv(source= i)

            # RESOLUTION PHASE
            # Start with even-even coords and end with odd-odd coords
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                signal_count = 0
                for processor_row in range(a, Classes.Grid.grid_index_limit, 2):
                    for processor_col in range(b, Classes.Grid.grid_index_limit, 2):
                        comm.send("proceed", get_rank(processor_row, processor_col))
                        signal_count += 1
                for _ in range(signal_count):
                    comm.recv()

            for i in range(1, n_ranks):
                comm.send("phase finished", i)

            for i in range(1,n_ranks):
                comm.recv(source= i)

            # HEALING PHASE
            # Start with even-even coords and end with odd-odd coords
            for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                signal_count = 0
                for processor_row in range(a, Classes.Grid.grid_index_limit, 2):
                    for processor_col in range(b, Classes.Grid.grid_index_limit, 2):
                        comm.send("proceed", get_rank(processor_row, processor_col))
                        signal_count += 1
                for _ in range(signal_count):
                    comm.recv()

            for i in range(1, n_ranks):
                comm.send("phase finished", i)




        # POST-WAVE UPDATES
        # Start with even-even coords and end with odd-odd coords
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            signal_count = 0
            for processor_row in range(a, Classes.Grid.grid_index_limit, 2):
                for processor_col in range(b, Classes.Grid.grid_index_limit, 2):
                    comm.send("proceed", get_rank(processor_row, processor_col))
                    signal_count += 1
            for _ in range(signal_count):
                comm.recv()
        for i in range(1, n_ranks):
            comm.send("next wave", i)

    print_array = [["." for _ in range(main_grid_size)] for _ in range(main_grid_size)]
    for i in range(1, n_ranks):
        status = MPI.Status()
        sub_grid = comm.recv(status=status)
        source_rank = status.Get_source()
        for r in sub_grid.units:
            for unit in r:
                if unit == ".":
                    continue
                x, y = get_absolute_coordinates(unit.x, unit.y, sub_grid_size, source_rank)
                if isinstance(unit, Classes.AirUnit):
                    print_array[x][y] = "A"
                elif isinstance(unit, Classes.WaterUnit):
                    print_array[x][y] = "W"
                elif isinstance(unit, Classes.FireUnit):
                    print_array[x][y] = "F"
                elif isinstance(unit, Classes.EarthUnit):
                    print_array[x][y] = "E"

    # Print the array with one space between elements
    for r in print_array:
        for element in r:
            output_file.write(element + " ")
            #print(element, end=" ")
        #print(flush=True)
        output_file.write("\n")



    input_file.close()
    output_file.close()


else:
    # Wait for the manager to calculate sub-grid size
    main_grid_size, sub_grid_size, wave_count, round_count = comm.recv(source=0)
    Classes.Grid.grid_index_limit = main_grid_size // sub_grid_size
    processor_row, processor_col = (rank-1) // Classes.Grid.grid_index_limit, (rank+1) % Classes.Grid.grid_index_limit
    grid = Classes.Grid(sub_grid_size, processor_row, processor_col)
    comm.send(grid, dest=0)
    for wave_number in range(wave_count):
        units = comm.recv(source=0)
        for unit in units:
            grid.create_unit(unit)
        comm.send("", dest=0)

        for round_number in range(round_count):
            movement_queue = []
            # MOVEMENT PHASE
            while True:
                # First wait for starting signal
                status = MPI.Status()
                signal = comm.recv(status=status)
                if signal == "proceed":
                    if not grid.has_airunit():
                        # If we don't have an air unit we are done in movement phase
                        comm.send("completed", dest=0)
                    else:
                        movement_queue = air_unit_movement()
                        comm.send("completed", dest=0)

                elif signal == "finish":
                    break

                elif signal[0] == "send unit type":
                    comm.send(type(grid.units[signal[1]][signal[2]]), status.Get_source())

                elif signal[0] == "send target count":
                    target_count = air_unit_target_count(signal[1], signal[2])
                    comm.send(target_count, status.Get_source())

            # Applying movements of movement phase
            # First, handle your own queue and send necessary signals to other workers
            for x, y, new_x, new_y in movement_queue:
                target_rank = rank + get_target_rank_offset(new_x, new_y)
                if target_rank == rank:
                    grid.add_unit(grid.remove_unit(grid.units[x][y]), new_x, new_y)
                else:
                    comm.send((grid.remove_unit(grid.units[x][y]), new_x % sub_grid_size, new_y % sub_grid_size), dest=target_rank)
            comm.send("finished queue", dest=0)



            # Now handle signals came from other processors
            while True:
                status = MPI.Status()
                signal = comm.recv(status=status)
                if status.Get_source() == 0:
                    comm.send("phase finished", dest=0)
                    break
                else:
                    grid.add_unit(signal[0], signal[1], signal[2])

            # Wait for every other processor is done with current phase
            comm.recv(source=0)  # wait for "phase finished"






            # ATTACK PHASE

            # First we need to fetch the attackers
            attackers = []
            while True:
                status = MPI.Status()
                signal = comm.recv(status=status)
                if signal == "proceed":
                    for r in grid.units:
                        for unit in r:
                            if unit == ".":
                                continue
                            # Those with low health will skip
                            elif unit.health < unit.max_health // 2:
                                unit.skip = True
                                continue
                            else:
                                relative_coordinates = unit.target_coordinates()
                                for [rel_x, rel_y,dir_x,dir_y] in relative_coordinates:
                                    # Get the rank offset for relative coordinates
                                    rank_offset = get_target_rank_offset(rel_x, rel_y)
                                    if rank_offset is None:
                                        continue
                                    # If target coordinates in our grid check whether valid target exists or not
                                    if rank_offset == 0 and grid.units[rel_x][rel_y] != "." and type(unit) != type(grid.units[rel_x][rel_y]):
                                        attackers.append(unit)
                                        unit.skip = False
                                        break
                                    # If an air unit encounters with a neutral cell it can aim the next cell
                                    elif rank_offset == 0 and grid.units[rel_x][rel_y] == "." and isinstance(unit, Classes.AirUnit):
                                        # Range can't exceed 2
                                        if dir_x == 2 or dir_y == 2 or dir_x == -2 or dir_y == -2 :
                                            continue
                                        else:
                                            # Add the next cell into the queue
                                            relative_coordinates.append([rel_x + dir_x, rel_y + dir_y, 2*dir_x, 2*dir_y])

                                    # If target coordinates are in another grid send signals
                                    elif rank_offset != 0:
                                        # request information of unit type
                                        comm.send((rel_x % sub_grid_size, rel_y % sub_grid_size), dest=rank + rank_offset)
                                        unit_type = comm.recv(source=rank + rank_offset)
                                        # If target cell is neutral air units will target next cell
                                        if unit_type == type(".") and isinstance(unit, Classes.AirUnit):
                                            if dir_x == 2 or dir_y == 2 or dir_x == -2 or dir_y == -2 :
                                                continue
                                            # Add the next cell into the queue
                                            relative_coordinates.append([rel_x + dir_x, rel_y + dir_y, 2 * dir_x, 2 * dir_y])
                                        # If target cell is an invalid target continue
                                        elif unit_type == type(unit) or unit_type == type("."):
                                            continue
                                        # If target cell is a valid target we got a new attacker
                                        else:
                                            attackers.append(unit)
                                            unit.skip = False
                                            break
                                    else:
                                        continue

                    comm.send("finished", dest=0)

                elif signal == "decisions finished":
                    break
                else:
                    # Handle data request from another process
                    (x, y) = signal
                    comm.send(type(grid.units[x][y]), status.Get_source())




            comm.send("In Resolution Phase", dest=0)
            # RESOLUTION PHASE
            # First deal the damages without killing anyone
            while True:
                status = MPI.Status()
                signal = comm.recv(status=status)
                # If proceed deal damage
                if signal == "proceed":
                    for attacker in attackers:
                        relative_coordinates = attacker.target_coordinates()
                        for [rel_x, rel_y, dir_x, dir_y] in relative_coordinates:
                            rank_offset = get_target_rank_offset(rel_x, rel_y)
                            # If target processor is invalid continue
                            if rank_offset is None:
                                continue
                            # If target coordinates in our grid check whether valid target exists or not
                            if rank_offset == 0 and grid.units[rel_x][rel_y] != "." and type(attacker) != type(grid.units[rel_x][rel_y]):
                                grid.units[rel_x][rel_y].take_damage(attacker.attack_power)
                            elif rank_offset == 0 and grid.units[rel_x][rel_y] == "." and isinstance(attacker,Classes.AirUnit):
                                # Range can't exceed 2
                                if dir_x == 2 or dir_y == 2 or dir_x == -2 or dir_y == -2:
                                    continue
                                else:
                                    # Add the next cell into the queue
                                    relative_coordinates.append([rel_x + dir_x, rel_y + dir_y, 2*dir_x,2*dir_y])


                            # If the target is not in our grid send a message to target processor
                            elif rank_offset != 0:
                                comm.send((rel_x % sub_grid_size, rel_y % sub_grid_size, attacker.attack_power,type(attacker),dir_x,dir_y), dest=rank + rank_offset)
                            else:
                                continue
                    comm.send("finished", dest=0)
                elif signal == "phase finished":
                    break
                else:
                    (x, y, damage, attacker_type,dir_x,dir_y) = signal
                    # If attacker is an air unit and target is neutral try next cell
                    if attacker_type == Classes.AirUnit and grid.units[x][y] == ".":
                        # Make sure range does not exceed 2
                        if not (dir_x == 2 or dir_y == 2 or dir_x == -2 or dir_y == -2):
                            rel_x, rel_y = x + dir_x, y + dir_y
                            dir_x *= 2
                            dir_y *= 2
                            rank_offset = get_target_rank_offset(rel_x,rel_y)
                            if rank_offset is None:
                                continue
                            # Next cell is in this grid, deal the damage
                            elif rank_offset == 0:
                                if not (grid.units[rel_x][rel_y] == "." or isinstance(grid.units[rel_x][rel_y],Classes.AirUnit)):
                                    grid.units[rel_x][rel_y].take_damage(damage)
                            # If the next cell is not in this grid send a new signal
                            else:
                                comm.send((rel_x % sub_grid_size, rel_y % sub_grid_size, damage, attacker_type, dir_x, dir_y), dest=rank + rank_offset)


                    elif attacker_type == type(grid.units[x][y]) or grid.units[x][y] == ".":
                        continue
                    else:
                        grid.units[x][y].take_damage(damage)

            comm.send("In healing phase", dest=0)

            # HEALING PHASE
            # Bury the dead ones, heal the alive skippers
            while True:
                status = MPI.Status()
                signal = comm.recv(status=status)
                # If proceed start burying dead ones
                if signal == "proceed":
                    for r in grid.units:
                        for unit in r:
                            # If no dead unit continue
                            if unit == ".":
                                continue
                            if unit.is_alive():
                                if unit.skip:
                                    unit.heal()
                                continue
                            # If current unit is dead and not a fire unit look at neighbours and detect fire units
                            if not isinstance(unit, Classes.FireUnit):
                                for [i, j] in Classes.FireUnit.attack_pattern:
                                    rel_x, rel_y = unit.x + i, unit.y + j
                                    rank_offset = get_target_rank_offset(rel_x, rel_y)
                                    if rank_offset is None:
                                        continue
                                    elif rank_offset == 0:
                                        if isinstance(grid.units[rel_x][rel_y], Classes.FireUnit):
                                            grid.units[rel_x][rel_y].inferno()
                                    else:
                                        comm.send((rel_x % sub_grid_size, rel_y % sub_grid_size), dest=rank + rank_offset)
                            grid.remove_unit(unit)
                    comm.send("finished", dest=0)
                elif signal == "phase finished":
                    break
                else:
                    (x, y) = signal
                    if isinstance(grid.units[x][y], Classes.FireUnit):
                        grid.units[x][y].inferno()

            # Reset round-temporary information
            for r in grid.units:
                for unit in r:
                    if unit == ".":
                        continue
                    unit.skip = True
                    if isinstance(unit, Classes.FireUnit):
                        unit.inferno_applied = False





        flood_queue = []
        # Apply post-wave updates
        while True:
            status = MPI.Status()
            signal = comm.recv(status=status)
            # If proceed start post-wave updates and handle flood
            if signal == "proceed":
                for r in grid.units:
                    for unit in r:
                        # Processes inferno damage reset
                        if isinstance(unit, Classes.FireUnit):
                            unit.attack_power = 4

                        # Processes flood
                        elif isinstance(unit, Classes.WaterUnit):
                            # Low x has priority
                            for [i, j] in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
                                rel_x, rel_y = unit.x + i, unit.y + j
                                rank_offset = get_target_rank_offset(rel_x, rel_y)
                                if rank_offset is None:
                                    continue
                                if rank_offset == 0:
                                    if grid.units[rel_x][rel_y] == ".":
                                        flood_queue.append([rel_x, rel_y])
                                        break
                                else:
                                    comm.send((rel_x % sub_grid_size, rel_y % sub_grid_size), dest=rank + rank_offset)
                                    add_success = comm.recv(source=rank+rank_offset)
                                    if add_success: break
                comm.send("updates finished", dest=0)

            elif signal == "next wave":
                break

            else:
                # Add the unit neighbour requested
                rel_x, rel_y = signal
                add_success = False
                if grid.units[rel_x][rel_y] == ".":
                    flood_queue.append([rel_x, rel_y])
                    add_success = True
                comm.send(add_success, status.Get_source())



        # Apply the changes in flood_queue
        for rel_x, rel_y in flood_queue:
            grid.create_unit(("W", rel_x, rel_y))

    comm.send(grid, dest=0)
