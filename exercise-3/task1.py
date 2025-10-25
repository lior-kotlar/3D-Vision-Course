import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json

# --- Global variables to store state ---
line_names = ['l', 'l_prime', 'l_double_prime']
point_names = ['p', 'p_prime', 'p_double_prime']
# We need to store the axis and figure to update them
fig = None
ax = None

class RankComputation():
    def __init__(self, data_directory, correspondence_type):
        self.data_directory = data_directory
        self.correspondence_type = correspondence_type
        self.triple_filename = f'{correspondence_type}_triple.json'
        self.correspondence_types = {
            'lll': [self.get_user_lll_and_save,
                    self.onclick_lll,
                    self.compute_rank_lll],
            'ppp': [self.get_user_ppp_and_save,
                    self.onclick_ppp,
                    self.compute_rank_ppp],
            'ppl': [self.get_user_ppl_and_save,
                    self.onclick_ppl,
                    self.compute_rank_ppl]
        }
        self.image_paths = self.get_image_paths()
        self.triple = self.get_points_from_file_if_exists()
        self.get_user_triple_function, self.onclick_function, self.compute_rank_function = self.configure_correspondence_type(correspondence_type)

    def get_points_from_file_if_exists(self):
        triple = []
        if os.path.isfile(os.path.join(self.data_directory, self.triple_filename)):
            print(f"\nLoading existing triple from {self.triple_filename}...")
            with open(os.path.join(self.data_directory, self.triple_filename), 'r') as f:
                data = json.load(f)
                triple = [
                    np.array(data[f"{self.correspondence_type[0]}"]),
                    np.array(data[f"{self.correspondence_type[1]}"]),
                    np.array(data[f"{self.correspondence_type[2]}"])
                ]
            print("Triple loaded successfully.")
            
        return triple

    def v_to_skew(self, v):
            vx, vy, vz = v
            skew_matrix = np.array([[0, -vz, vy],
                                    [vz, 0, -vx],
                                    [-vy, vx, 0]])
            return skew_matrix
    
    def configure_correspondence_type(self, correspondence_type):
        attributes = (self.correspondence_types.get(correspondence_type))
        run_function, onclick_function, rank_computation_function = attributes
        return run_function, onclick_function, rank_computation_function

    def compute_rank_lll(self):
        l = self.triple[0].flatten()
        l_prime = self.triple[1].flatten()
        l_double_prime = self.triple[2].flatten()
        lx, ly, lz = l
        l_prime_t = l_prime.reshape(1, 3)
        l_double_prime_t = l_double_prime.reshape(1, 3)

        a_t = np.kron(l_double_prime_t, l_prime_t)
        zero_block = np.zeros((1, 9))

        row1 = np.hstack([zero_block, lz*a_t, -ly*a_t])
        row2 = np.hstack([-lz*a_t, zero_block, lx*a_t])
        row3 = np.hstack([ly*a_t, -lx*a_t, zero_block])

        m = np.vstack([row1, row2, row3])
        rank = np.linalg.matrix_rank(m)

        return rank

    def compute_rank_ppp(self):
        
        p1, p2, p3 = self.triple
        p1x, p1y, p1z = p1

        p2_skew = self.v_to_skew(p2)
        p3_skew_t = self.v_to_skew(p3).T

        k_matrix = np.kron(p2_skew, p3_skew_t) 

        i_9 = np.eye(9)
        p1_as_matrices = np.hstack([p1x * i_9, p1y * i_9, p1z * i_9])
        m = k_matrix @ p1_as_matrices
        rank = np.linalg.matrix_rank(m)
        return rank

    def compute_rank_ppl(self):
        x, x_prime, l_double_prime = self.triple
        x1, x2, x3 = x.flatten()
    
    # 2. Create the known 3x3 matrix L
        L = self.v_to_skew(x_prime)
        
        # 3. Create the 3x9 matrix R = (l''^T ⊗ I_3)
        R = np.kron(l_double_prime.reshape(1, 3), np.eye(3))
        
        # 4. Create the 3x9 base matrix B = L @ R
        B = L @ R
        
        # 5. Build the final 3x27 matrix A = [x1*B, x2*B, x3*B]
        A = np.hstack([x1 * B, x2 * B, x3 * B])

        rank = np.linalg.matrix_rank(A)
        return rank

    def run(self):
        if len(self.triple) == 0:
            self.get_user_triple_function()
        if len(self.triple) == 3:
            rank = self.compute_rank_function()
            print(f"\nComputed rank of the matrix: {rank}")
        else:
            print("Error: Triple is incomplete. Cannot compute rank.")

    def save_triple_to_file(self):
        """Saves the calculated line equations to a JSON file."""
        filename = f'{self.correspondence_type}_triple.json'
        output_path = os.path.join(self.data_directory, filename)
        print(f"\nSaving to {filename}...")
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    f"{self.correspondence_type[0]}": list(self.triple[0].tolist()),
                    f"{self.correspondence_type[1]}": list(self.triple[1].tolist()),
                    f"{self.correspondence_type[2]}": list(self.triple[2].tolist())
                }, f)
            print("File saved successfully.")
        except IOError as e:
            print(f"Error saving file: {e}")

    # def save_points_to_file(self):
    #     """Saves the calculated points to a JSON file."""
    #     filename = 'ppp.json'
    #     output_path = os.path.join(self.data_directory, filename)
    #     print(f"\nSaving 3 points to {filename}...")
    #     try:
    #         with open(output_path, 'w') as f:
    #             json.dump({
    #                 "p": list(self.triple[0].tolist()),
    #                 "p_prime": list(self.triple[1].tolist()),
    #                 "p_double_prime": list(self.triple[2].tolist())
    #             }, f)
    #         print("File saved successfully.")
    #     except IOError as e:
    #         print(f"Error saving file: {e}")

    def onclick_lll(self, event, plot_state):
        """
        Main event handler for mouse clicks.
        'plot_state' is a dictionary containing state for *this plot only*.
        """
        # Unpack the state for this specific plot
        fig = plot_state["fig"]
        ax = plot_state["ax"]
        line_name = plot_state["line_name"]
        cid = plot_state["cid"]

        # Ignore clicks outside the plot axes
        if not event.inaxes:
            return

        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Add click as a homogeneous point
        p_homogeneous = np.array([x, y, 1.0])
        plot_state["points"].append(p_homogeneous)
        
        # print(f"  Point {len(plot_state['points'])} clicked at: ({x:.2f}, {y:.2f})")
        
        # Plot a marker at the click location
        ax.plot(x, y, 'rx', markersize=8)

        if len(plot_state["points"]) == 1:
            # Only one point clicked so far
            ax.set_title(f"Click point 2 for line '{line_name}'")
        
        elif len(plot_state["points"]) == 2:
            # This is the second click, we can define the line
            p1 = plot_state["points"][0]
            p2 = plot_state["points"][1]
            
            # Calculate the line using the cross product
            line_eq = np.cross(p1, p2)
            
            # Normalize the line equation
            norm_factor = np.sqrt(line_eq[0]**2 + line_eq[1]**2)
            if norm_factor > 1e-6:
                line_eq = line_eq / norm_factor
            
            # print(f"  Line '{line_name}' calculated: {line_eq}")
            
            # Add the result to our *global* list
            self.triple.append(line_eq)
                        
            # Update title
            ax.set_title(f"Line '{line_name}' defined. Close this window to continue.")
            
            # We are done with this plot. Disconnect the event handler
            # and close the window.
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig) # This un-blocks the 'plt.show()' call in main()
        
        # Redraw the plot
        fig.canvas.draw()

    def onclick_ppp(self, event, plot_state):
        """
        Main event handler for mouse clicks.
        'plot_state' is a dictionary containing state for *this plot only*.
        """
        # Unpack the state for this specific plot
        fig = plot_state["fig"]
        ax = plot_state["ax"]
        point_name = plot_state["point_name"]
        cid = plot_state["cid"]

        # Ignore clicks outside the plot axes
        if not event.inaxes:
            return

        # --- This is the first and only click ---
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        
        # Create homogeneous point
        p_homogeneous = np.array([x, y, 1.0], dtype=np.int32)
        
        print(f"  Point '{point_name}' clicked at: ({x:.2f}, {y:.2f})")
        
        # Add the result to our *global* list
        self.triple.append(p_homogeneous)
            
        # Update title
        ax.set_title(f"Point '{point_name}' defined. Close this window to continue.")
        
        # We are done with this plot. Disconnect the event handler
        # and close the window.
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig) # This un-blocks the 'plt.show()' call in main()
        
        # Redraw the plot (briefly, before it closes)
        fig.canvas.draw()

    def onclick_ppl(self, event, plot_state):
        """
        Main event handler for mouse clicks.
        Handles both single-click (for points) and double-click (for lines) logic.
        """
        # Unpack the state for this specific plot
        fig = plot_state["fig"]
        ax = plot_state["ax"]
        item_type = plot_state["item_type"]
        item_name = plot_state["item_name"]
        cid = plot_state["cid"]

        if not event.inaxes: return

        # Get click coordinates and store as homogeneous point
        x, y = event.xdata, event.ydata
        p_homogeneous = np.array([x, y, 1.0])
        plot_state["points"].append(p_homogeneous)
        
        print(f"  Click {len(plot_state['points'])} at: ({x:.2f}, {y:.2f})")

        if item_type == 'point':
            # --- This is a POINT image, we only need one click ---
            
            # Add the result to our *global* list
            self.triple.append(p_homogeneous)
            
            # Update title and close
            ax.set_title(f"Point '{item_name}' defined. Close this window.")
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            
        elif item_type == 'line':
            # --- This is a LINE image, we need two clicks ---
            
            # Plot a marker at the click location
            ax.plot(x, y, 'rx', markersize=8)

            if len(plot_state["points"]) == 1:
                # First of two clicks
                ax.set_title(f"Click point 2 for line '{item_name}'")
            
            elif len(plot_state["points"]) == 2:
                # Second click, we can define the line
                p1, p2 = plot_state["points"]
                
                line_eq = np.cross(p1, p2)
                
                # Normalize
                norm_factor = np.sqrt(line_eq[0]**2 + line_eq[1]**2)
                if norm_factor > 1e-6:
                    line_eq = line_eq / norm_factor
                
                print(f"  Line '{item_name}' calculated: {line_eq}")
                
                # Add the result to our *global* list
                self.triple.append(line_eq)
                            
                # Update title and close
                ax.set_title(f"Line '{item_name}' defined. Close this window.")
                fig.canvas.mpl_disconnect(cid)
                plt.close(fig)

        # Redraw the plot
        fig.canvas.draw()

    def get_image_paths(self):
        if not os.path.isdir(self.data_directory):
            print(f"Error: Directory '{self.data_directory}' does not exist.")
            exit(1)
        image_paths = os.listdir(self.data_directory)
        image_paths = [os.path.join(self.data_directory, p) for p in image_paths if p.endswith(('.png', '.jpg', '.jpeg'))]
        if len(image_paths) != 3:
            print("Error: Exactly 3 image paths must be in specified directory.")
            exit()
        return image_paths

    def get_lll_from_user(self):
        global fig, ax
        
        for line_name, image_path in zip(line_names, self.image_paths):
            try:
                image = plt.imread(image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}'")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading image: {e}")
                sys.exit(1)
            image_name = os.path.basename(image_path)
            print(f"\nProcessing line '{line_name}' from image '{image_name}'...")
            # 3. Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_aspect('equal')
            ax.set_title(f"Click point 1 for line '{line_name}'")
            ax.set_xlabel("Click 2 points to define the line. Do not zoom or pan.")
            
            # 4. Create a state dictionary for this plot
            # This avoids complex globals and keeps each plot's state separate.
            plot_state = {
                "points": [],
                "line_name": line_name,
                "fig": fig,
                "ax": ax,
                "cid": None # Will store the connection ID
            }

            # 5. Connect the click event handler
            # We use a lambda function to pass our 'plot_state' to the handler
            cid = fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: self.onclick_function(event, plot_state)
            )
            plot_state["cid"] = cid # Store the ID so we can disconnect it later

            # 6. Show the plot.
            # This call is BLOCKING. The script will pause here until
            # 'plt.close(fig)' is called (inside our onclick handler).
            plt.show() 
            
            print(f"Finished processing line '{line_name}'.")
        
    def get_user_lll_and_save(self):
        self.get_lll_from_user()
        self.save_triple_to_file()

    def get_ppp_from_user(self):
        global fig, ax
        
        for point_name, image_path in zip(point_names, self.image_paths):
            try:
                image = plt.imread(image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}'")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading image: {e}")
                sys.exit(1)
            image_name = os.path.basename(image_path)
            print(f"\nProcessing point '{point_name}' from image '{image_name}'...")
            # 3. Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_aspect('equal')
            ax.set_title(f"Click point 1 for point '{point_name}'")
            
            # 4. Create a state dictionary for this plot
            # This avoids complex globals and keeps each plot's state separate.
            plot_state = {
                "points": [],
                "point_name": point_name,
                "fig": fig,
                "ax": ax,
                "cid": None # Will store the connection ID
            }

            # 5. Connect the click event handler
            # We use a lambda function to pass our 'plot_state' to the handler
            cid = fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: self.onclick_function(event, plot_state)
            )
            plot_state["cid"] = cid # Store the ID so we can disconnect it later

            # 6. Show the plot.
            # This call is BLOCKING. The script will pause here until
            # 'plt.close(fig)' is called (inside our onclick handler).
            plt.show() 

            print(f"Finished processing point '{point_name}'.")

    def get_user_ppp_and_save(self):
        self.get_ppp_from_user()
        self.save_triple_to_file()

    def get_ppl_from_user(self):
        items_to_collect = [
            ('point', 'p'),         # Item 1: point 'p'
            ('point', 'p_prime'),    # Item 2: point 'p_prime'
            ('line',  'l_double_prime') # Item 3: line 'l_double_prime'
        ]
        

        print("--- Point-Point-Line Correspondence Selector ---")

        # 2. Loop through each image and collect the required data
        for i, (item_type, item_name) in enumerate(items_to_collect):
            image_path = self.image_paths[i]
            try:
                image = plt.imread(image_path)
            except FileNotFoundError:
                print(f"Error: Image file not found at '{image_path}'")
                sys.exit(1)
            except Exception as e:
                print(f"Error loading image: {e}")
                sys.exit(1)
            image_name = os.path.basename(image_path)

            print(f"\nProcessing image for {item_type} '{item_name}' ({image_name})...")

            # 3. Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_aspect('equal')
            
            # Set initial title and instructions
            if item_type == 'point':
                ax.set_title(f"Click one point for '{item_name}'")
                ax.set_xlabel("Click 1 point. Do not zoom or pan.")
            else: # item_type == 'line'
                ax.set_title(f"Click point 1 for line '{item_name}'")
                ax.set_xlabel("Click 2 points. Do not zoom or pan.")
            
            # 4. Create state dictionary for this plot
            plot_state = {
                "points": [],
                "item_type": item_type,
                "item_name": item_name,
                "fig": fig,
                "ax": ax,
                "cid": None
            }

            # 5. Connect the click event handler
            cid = fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: self.onclick_ppl(event, plot_state)
            )
            plot_state["cid"] = cid 

            # 6. Show the plot (blocking)
            plt.show() 
            
            print(f"Finished processing {item_type} '{item_name}'.")

    def get_user_ppl_and_save(self):
        self.get_ppl_from_user()
        self.save_triple_to_file()

def main():
    if len(sys.argv) < 3:
        print("Usage: python <script_name.py> <correspondence_type> <directory_path>")
        sys.exit(1)
    correspondence_type = sys.argv[1]
    directory_path = sys.argv[2]
    rank_computation = RankComputation(data_directory=directory_path, correspondence_type=correspondence_type)
    rank_computation.run()

if __name__ == "__main__":
    main()