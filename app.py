from flask import Flask, request, jsonify, send_from_directory
import perudo
import os
import dqn

app = Flask(__name__, static_folder='html')

# Load policy map from disk
perudo.POLICY_MAP = perudo.load_policy()

# Initialize dqn player
dqnplayer = dqn.DQNPlayer("DQNPlayer")
checkpoint = dqn.torch.load("dqnmodel.dat")
dqnplayer.model.load_state_dict(checkpoint["model_state_dict"])

# Instantiate global RLPlayer
rl_player = perudo.RLPlayer("RL-Agent")

@app.route('/policy', methods=['GET'])
def get_policy():
    """
    Get policy action for a given state vector.
    Expects GET parameters: state (comma-separated list of 9 integers)
    Returns: JSON with action [quantity, face] or [-1, -1] for doubt
    """
    state_param = request.args.get('state')
    if not state_param:
        return jsonify({'error': 'Missing state parameter'}), 400
    
    try:
        # Parse state vector from comma-separated string
        state_list = [int(x.strip()) for x in state_param.split(',')]
        
        # Validate state vector length (should be 9: total_dices, last_bid_q, last_bid_f, 6 dice counts)
        if len(state_list) != 9:
            return jsonify({'error': f'State vector must have 9 elements, got {len(state_list)}'}), 400
        
        # Convert to tuple as expected by policy function
        state = tuple(state_list)
        
        # Call policy with epsilon=0.01
        action = perudo.policy(state, epsilon=0.01)
        
        # Return action as JSON
        return jsonify({
            'action': list(action),
            'quantity': action[0],
            'face': action[1]
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid state format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/dqn_policy', methods=['GET'])
def dqn_policy():
    """
    Get policy action for a given state vector.
    Expects GET parameters: state (comma-separated list of 9 integers)
    Returns: JSON with action [quantity, face] or [-1, -1] for doubt
    """
    state_param = request.args.get('state')
    if not state_param:
        return jsonify({'error': 'Missing state parameter'}), 400
    
    try:
        # Parse state vector from comma-separated string
        state_list = [int(x.strip()) for x in state_param.split(',')]
        
        # Validate state vector length (should be 9: total_dices, last_bid_q, last_bid_f, 6 dice counts)
        if len(state_list) != 9:
            return jsonify({'error': f'State vector must have 9 elements, got {len(state_list)}'}), 400
        
        # Convert to tuple as expected by policy function
        state = tuple(state_list)
        
        # Call policy with epsilon=0.01
        action = perudo.n_to_action(dqnplayer.predict_action(state))
        
        # Return action as JSON
        return jsonify({
            'action': list(action),
            'quantity': action[0],
            'face': action[1]
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid state format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    """
    Serve static files from the html/ folder.
    If path is empty, serve index.html (or perudo.html if it exists).
    """
    if not path:
        # Try to serve perudo.html as default
        if os.path.exists('html/perudo.html'):
            return send_from_directory('html', 'perudo.html')
        return jsonify({'error': 'No index file found'}), 404
    
    # Serve the requested file from html/ folder
    return send_from_directory('html', path)

# Note:
# Flask's debug=True enables the interactive debugger and auto-reload, which can expose sensitive information and allow code execution.
# It is dangerous when the server is exposed publicly.
# We set debug=False and host='0.0.0.0' to listen on all interfaces safely.

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
